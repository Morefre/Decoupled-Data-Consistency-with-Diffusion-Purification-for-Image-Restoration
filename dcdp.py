from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from  torch.cuda.amp import autocast
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.logger import get_logger

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compare_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

def get_lpips(img1, img2, lpips, device):
  '''
  img1: torch.tensor of shape [1,C,H,W]
  img2: torch.tensor of shape [1,C,H,W]
  '''
  # Evaluate the lpips on device
  lpips.to(device)
  img1 = torch.clamp(img1, min=-1, max=1).to(device)
  img2 = torch.clamp(img2, min=-1, max=1).to(device)
  return lpips(img1, img2).detach().cpu().numpy()

def torch_to_np(img_torch):
  '''
  img_torch: torch.tensor of shape [1,C,H,W]
  '''
  img_np = img_torch[0].permute(1,2,0).detach().cpu().numpy()
  return img_np

def normalize_image(image):
  image = image-torch.min(image)
  image = image/torch.max(image)
  return image

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def Purification_Schedule(num_purification_steps, initial_timestep, end_timestep=0, schedule_type='linear'):
  '''
  Time schedule used for the diffusion purification process. 
  The results in our paper are all based on a linar schedule, but we implement other types of schedule here.
  '''
  assert num_purification_steps <= initial_timestep
  if schedule_type == 'constant':
    timesteps = num_purification_steps*[initial_timestep]
  elif schedule_type == 'linear':
    timesteps = np.linspace(0,1,num_purification_steps)*(initial_timestep-end_timestep)
    timesteps = timesteps + 1e-6
    timesteps = timesteps.round().astype(np.int64)
    timesteps = np.flip(timesteps+end_timestep)
    timesteps[timesteps==0] = 1
  elif schedule_type == 'cosine':
    timesteps = np.linspace(0,1,num_purification_steps)
    timesteps = timesteps*np.pi/2
    timesteps = np.cos(timesteps)**2
    timesteps = timesteps*(initial_timestep-end_timestep)
    timesteps = timesteps.round().astype(np.int64)
    timesteps = timesteps + end_timestep
    timesteps[timesteps==0] = 1
  elif schedule_type == 'bias_t1':
    timesteps = np.linspace(0,1,num_purification_steps)
    timesteps = timesteps*np.pi/2
    timesteps = np.sin(timesteps)
    timesteps = timesteps*(initial_timestep-end_timestep)
    timesteps = timesteps.round().astype(np.int64)
    timesteps = np.flip(timesteps+end_timestep)
    timesteps[timesteps==0] = 1
  elif schedule_type == 'bias_t0':
    timesteps = np.linspace(0,1,num_purification_steps)
    timesteps = timesteps-1
    timesteps = np.sin(timesteps*np.pi/2)+1
    timesteps = timesteps*(initial_timestep-end_timestep)
    timesteps = timesteps.round().astype(np.int64)
    timesteps = np.flip(timesteps+end_timestep)
    timesteps[timesteps==0] = 1
  elif schedule_type == 'reverse_cosine':
    linear_timesteps = Purification_Schedule(num_purification_steps, initial_timestep, end_timestep=end_timestep, schedule_type='linear')
    cosine_timesteps = Purification_Schedule(num_purification_steps, initial_timestep, end_timestep=end_timestep, schedule_type='cosine')
    timesteps = 2*linear_timesteps - cosine_timesteps
    timesteps[timesteps==0] = 1
  return timesteps

def CSGM_Solver_Pixel_Space(measurements, x_init, num_iterations, device, operator, use_weight_decay=False, weight_decay_lambda=0, 
                            mask=None, optimizer = 'SGD', momentum=0.9, type='L2', lr=0.1, save_every=50, verbose=False):
  '''
  This is the solver for the data fidelity optimization problem: 1/2||A(x)-y||_2^2 + weight_decay*||x-x_k||
  x_init: initial point x_k
  mask: inpainting mask
  '''
  if mask != None:
    mask = mask.to(device)

  x = x_init.clone().detach().requires_grad_(True)
  if optimizer == 'Adam':
    optimizer = torch.optim.Adam([x],lr=lr)
  elif optimizer == 'SGD':
    # Momentum accelerates the reconstruction speed, which ususally leads to better results (0.9)
    optimizer = torch.optim.SGD([x],lr=lr,momentum=momentum)
  if type == 'L2':
    criterion = torch.nn.MSELoss().to(device)
  elif type == 'L1':
    criterion = torch.nn.L1Loss().to(device)
  x_list = []

  measurements = measurements.clone().detach()

  for i in range(num_iterations):
    optimizer.zero_grad()
    recon = x
    if mask != None:
      recon_loss = criterion(measurements, operator.forward(recon,mask=mask))
    else:
      recon_loss = criterion(measurements, operator.forward(recon))
    if use_weight_decay == True:
      weight_decay_loss = criterion(x, x_init)
      recon_loss = recon_loss + weight_decay_lambda*weight_decay_loss
    recon = normalize_image(recon)
    recon_loss.backward()
    optimizer.step()

    if i % save_every == 0 or i == num_iterations-1:
      x_list.append(x.clone().detach())
      if verbose == True:
        plt.figure()
        plt.title('Iter: '+str(i+1))
        plt.imshow(torch_to_np(recon))
        plt.show()
    
  return x_list[-1], x_list

def Diffusion_Purified_CSGM(model, img_gt, total_num_iterations, csgm_num_iterations, device, cond_method,
                            ddim_init_timestep, ddim_end_timestep, operator, inverse_problem_type, noise_std, 
                            use_weight_decay=False, weight_decay_lambda=0, mask=None, full_ddim = True, 
                            ddim_num_iterations=20, purification_schedule='linear', optimizer='Adam', 
                            momentum=0, lr=0.1, save_every_main=50, save_every_sub=1, 
                            verbose=False, root_path=None):
    '''
    model: The pretrained diffusion model
    img_gt: ground truth image
    total_num_iterations: total number of iterations (K) of the algorithm
    csgm_num_iterations: number of gradient steps used to solve the data fidelity optimization sub-problem
    ddim_init_timestep: T_0 in the paper
    ddim_end_timestep: T_K at the final iteration
    purification_schedule: A decaying schedule from T_0 to T_k
    '''
    if mask != None:
        mask = mask.to(device)
    img_gt = img_gt.to(device)

    model = model.to(device)
    model.eval()

    # If doing nonlinear deblurring, we generate different random kernels for each image
    if inverse_problem_type == 'nonlinear_blur':
       random_kernel = torch.randn(1, 512, 2, 2).to(device) * 1.2
       operator.random_kernel = random_kernel

    # Create and save noisy measurements
    if mask != None:
        measurements = operator.forward(img_gt,mask=mask)
    else:
        measurements = operator.forward(img_gt)
  
    measurements = measurements+torch.randn(measurements.shape).to(device)*noise_std
    
    plt.figure()
    plt.imshow(torch_to_np(normalize_image(measurements)))
    plt.title('Measurements')
    plt.savefig(root_path+'measurements.png')

    x = torch.zeros(img_gt.shape, device=device, requires_grad=True)
    x_list_complete = []
    # Initialize the purification timesteps
    purification_timesteps = Purification_Schedule(total_num_iterations, ddim_init_timestep, ddim_end_timestep, schedule_type=purification_schedule)

    # The base sampler is responsible for running the forward process
    base_diffusion = create_sampler(sampler='ddpm',
                                    steps = 1000,
                                    noise_schedule='linear',
                                    model_mean_type='epsilon',
                                    model_var_type='learned_range',
                                    dynamic_threshold=False,
                                    clip_denoised=True,
                                    rescale_timesteps=False,
                                    timestep_respacing=1000)
    for i in range(total_num_iterations):
        ddim_timestep = purification_timesteps[i]

        # Step 1: Perform data fidelity optimization with graidient descent (csgm)
        x, x_list_sub = CSGM_Solver_Pixel_Space(measurements, x, csgm_num_iterations,
                                    device, use_weight_decay=use_weight_decay, 
                                    weight_decay_lambda=weight_decay_lambda, 
                                    operator=operator, mask=mask, optimizer=optimizer, 
                                    momentum=momentum, lr=lr, save_every=save_every_sub)

        # Step 2: Purify the current x with the pretraind diffusion model
        
        # Add Noise to the current x
        x_noisy = base_diffusion.q_sample(x, ddim_timestep-1)

        # Purification
        # Create DDIM Sampler
        if ddim_timestep == 1:
          ddim_timestep = ddim_timestep + 1
        if full_ddim == True:
            if ddim_timestep <= ddim_num_iterations:
                ddim_num_iters = ddim_timestep
            else:
                ddim_num_iters = ddim_num_iterations
            ddim_num_iters = int(ddim_num_iters)
            DDIM_Sampler = create_sampler(sampler='ddim', 
                                          steps=ddim_timestep, 
                                          noise_schedule='linear',
                                          model_mean_type='epsilon',
                                          model_var_type='learned_range',
                                          dynamic_threshold=False,
                                          clip_denoised=True,
                                          rescale_timesteps=False,
                                          timestep_respacing=ddim_num_iters
                                         )
            measurement_cond_fn = cond_method.conditioning
            # Runing the sampling reverse process
            sample_fn = partial(DDIM_Sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
            
            if inverse_problem_type == 'inpainting':
                measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
                sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
            
            with autocast():
                x_purified, _ = sample_fn(x_start=x_noisy, measurement=measurements, record=False, save_root=None)
        
        # Version 2 of the algorithm, instead of performing reverse process, we directly use Tweedie's formula for one step estimation
        elif full_ddim == False:
            sample_fn = partial(base_diffusion.p_sample, model=model)
            with autocast():
              out = sample_fn(x=x_noisy, t=torch.tensor(int(ddim_timestep-1)).unsqueeze(0).to(device))
              x_purified = out['pred_xstart']
              x_purified = x_purified.detach()

        x_prev = x.clone().detach()
        x = x_purified
        x_list_complete = x_list_complete + x_list_sub
        x_list_complete.append(x)

        if i % save_every_main == 0 or i==total_num_iterations-1:
            if verbose == True:
                plt.figure(figsize=(40,10))
                plt.subplot(141)
                plt.title('gt x')
                plt.imshow(torch_to_np(normalize_image(img_gt)))
                plt.subplot(142)
                plt.title('x')
                plt.imshow(torch_to_np(normalize_image(x_prev)))
                plt.subplot(143)
                plt.title('x_noisy')
                plt.imshow(torch_to_np(normalize_image(x_noisy)))
                plt.subplot(144)
                plt.title('x_purified')
                plt.imshow(torch_to_np(normalize_image(x_purified)))
                fig_name = 'Iter_'+str(i)+'.png'
                plt.savefig(root_path+fig_name)
    return x, x_list_complete


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_config', type=str)
  parser.add_argument('--task_config', type=str)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--save_dir', type=str, default='./purification_results')
  parser.add_argument('--purification_config', type=str)
  args = parser.parse_args()

  # logger
  logger = get_logger()

  # Device setting
  device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
  logger.info(f"Device set to {device_str}.")
  device = torch.device(device_str)  

  # Load configurations
  model_config = load_yaml(args.model_config)
  task_config = load_yaml(args.task_config)
  purification_config = load_yaml(args.purification_config)
  
  # Load model
  model = create_model(**model_config)
  model = model.to(device)
  model.eval()
  
  # Prepare Operator and noise
  measure_config = task_config['measurement']
  operator = get_operator(device=device, **measure_config['operator'])
  noiser = get_noise(**measure_config['noise'])
  logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

  # Prepare conditioning method, this should always be set as 'vanilla'
  cond_method_name = 'vanilla'
  scale = 0

  # Set to False if use Version 2 of the algorithm   
  full_ddim = purification_config['purification']['full_ddim']

  cond_method = get_conditioning_method(cond_method_name, operator, noiser, scale=scale)
  logger.info(f"Conditioning method : {cond_method_name}")

  # Working directory
  out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
  os.makedirs(out_path, exist_ok=True)

  img_size = purification_config['others']['img_size']
  dataset_name = purification_config['dataset']['name']
  noise_std = measure_config['noise']['sigma']

  # Build dataset
  data_config = purification_config['dataset']
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  dataset = get_dataset(**data_config, transforms=transform)
  loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)


  inverse_problem_type = measure_config['operator']['name']
  if inverse_problem_type == 'inpainting':
      mask = torch.ones(img_size)
      mask[:,:,100:200,100:200] = 0
      mask = mask.to(device)
  else:
      mask = None

  PSNR_list_All = []
  SSIM_list_All = []
  LPIPS_list_All = []
  
  # Parameters for the diffusion Purification
  total_num_iterations = purification_config['purification']['total_num_iterations']
  csgm_num_iterations = purification_config['purification']['csgm_num_iterations']
  ddim_init_timestep = purification_config['purification']['ddim_init_timestep']
  ddim_end_timestep = purification_config['purification']['ddim_end_timestep']
  purification_schedule = purification_config['purification']['purification_schedule']
  ddim_num_iterations = purification_config['purification']['ddim_num_iterations']
  save_every_main = purification_config['purification']['save_every_main']
  save_every_sub = purification_config['purification']['save_every_sub']
  optimizer = purification_config['purification']['optimizer']
  lr = purification_config['purification']['lr']
  momentum = purification_config['purification']['momentum']
  full_ddim = purification_config['purification']['full_ddim']
  use_weight_decay = purification_config['purification']['use_weight_decay']
  weight_decay_lambda = purification_config['purification']['weight_decay_lambda']


  if use_weight_decay == False:
      weight_decay_lambda = 0
  path_0 = os.path.join(out_path, dataset_name, 'noise_std_'+str(noise_std), str(ddim_init_timestep)+'_'+str(ddim_end_timestep)+'_'+str(total_num_iterations)+'_'+str(csgm_num_iterations)+'_'+purification_schedule+'_'+str(lr)+'_'+str(momentum)+'_'+str(full_ddim)+'_'+str(ddim_num_iterations)+'_'+str(weight_decay_lambda))

  for i, img in enumerate(loader):
      if i>= 10: #Modify this line if you want to test on more images
         break
      img = img.to(device)
      root_path = path_0 + '/img_' + str(i) + '/'
      isExist = os.path.exists(root_path)
      if not isExist:
          os.makedirs(root_path)
      figure_root_path = root_path + 'figures/'
      isExist = os.path.exists(figure_root_path)
      if not isExist:
          # Create a new directory if it does not exist
          os.makedirs(figure_root_path)        
      x, x_list_complete = Diffusion_Purified_CSGM(model, img_gt=img, total_num_iterations=total_num_iterations, 
                                                      csgm_num_iterations=csgm_num_iterations, device=device,
                                                      cond_method=cond_method, ddim_init_timestep=ddim_init_timestep,
                                                      ddim_end_timestep=ddim_end_timestep, operator=operator,
                                                      inverse_problem_type=inverse_problem_type, noise_std=noise_std,
                                                      use_weight_decay=use_weight_decay, weight_decay_lambda=weight_decay_lambda,
                                                      mask=mask, full_ddim=full_ddim, ddim_num_iterations=ddim_num_iterations,
                                                      purification_schedule=purification_schedule, optimizer=optimizer,
                                                      momentum=momentum, lr=lr, save_every_main=save_every_main,
                                                      save_every_sub=save_every_sub, verbose=True, root_path=figure_root_path)
      
      # Save the intermediate reconstructions
      torch.save(x_list_complete,root_path+'x_list_complete.pt')

      img_np = torch_to_np(img)
      img_np = np.clip(img_np,-1,1)
      img_np = (img_np+1)/2
      PSNR_list = []
      SSIM_list = []
      LPIPS_list = []
      # Here we calculate the standard metrics on every intermediate reconstrucitons, which is time-costly. Comment the lines below if you want to reconstruct the images at a faster speed.
      for j in range(len(x_list_complete)):
        x = x_list_complete[j]
        recon = x.detach().cpu()
        recon_np = recon[0].permute(1,2,0).numpy()
        recon_np = np.clip(recon_np,-1,1)
        recon_np = (recon_np+1)/2
        PSNR_list.append(peak_signal_noise_ratio(img_np,recon_np))
        SSIM_list.append(compare_ssim(img_np, recon_np, channel_axis=2, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        LPIPS_list.append(get_lpips(img,recon,lpips=lpips,device=device))

      # Visualize PSNR, SSIM and LPIPS
      plt.figure(figsize=(30,10))
      plt.subplot(131)
      plt.plot(PSNR_list)
      plt.xlabel('Iteration/10')
      plt.title('CSGM Results for '+ inverse_problem_type+' (PSNR)')

      plt.subplot(132)
      plt.plot(SSIM_list)
      plt.xlabel('Iteration/10')
      plt.title('CSGM Results for '+ inverse_problem_type+' (SSIM)')

      plt.subplot(133)
      plt.plot(LPIPS_list)
      plt.xlabel('Iteration/10')
      plt.title('CSGM Results for '+ inverse_problem_type+' (LPIPS)')
      plt.savefig(figure_root_path+'metrics.png')
      
      print('Final PSNR: ',PSNR_list[-1],'Final SSIM: ',SSIM_list[-1],'Final LPIPS: ',LPIPS_list[-1])

      PSNR_list = np.array(PSNR_list)
      SSIM_list = np.array(SSIM_list)
      LPIPS_list = np.array(LPIPS_list)
      torch.save(PSNR_list, root_path+'/PSNR_list.pt')
      torch.save(SSIM_list, root_path+'/SSIM_list.pt')
      torch.save(LPIPS_list, root_path+'/LPIPS_list.pt')

      PSNR_list_All.append(PSNR_list)
      SSIM_list_All.append(SSIM_list)
      LPIPS_list_All.append(LPIPS_list)
  PSNR_list_All = np.array(PSNR_list_All)
  SSIM_list_All = np.array(SSIM_list_All)
  LPIPS_list_All = np.array(LPIPS_list_All)

  avg_PSNR_list = np.mean(PSNR_list_All, axis=0)
  std_PSNR_list = np.std(PSNR_list_All, axis=0)

  avg_SSIM_list = np.mean(SSIM_list_All, axis=0)
  std_SSIM_list = np.std(SSIM_list_All, axis=0)

  avg_LPIPS_list = np.mean(LPIPS_list_All, axis=0)
  std_LPIPS_list = np.mean(LPIPS_list_All, axis=0)

  print('Final average PSNR: ',avg_PSNR_list[-1],'Final average SSIM: ', avg_SSIM_list[-1],'Final average LPIPS: ',avg_LPIPS_list[-1])
  print('Final std PSNR: ',std_PSNR_list[-1],'Final std SSIM: ', std_SSIM_list[-1],'Final std LPIPS: ',std_LPIPS_list[-1])

  plt.figure(figsize=(30,10))
  plt.subplot(131)
  plt.plot(avg_PSNR_list)
  plt.xlabel('Iteration')
  plt.title('PSNR')

  plt.subplot(132)
  plt.plot(avg_SSIM_list)
  plt.xlabel('Iteration')
  plt.title('SSIM')

  plt.subplot(133)
  plt.plot(avg_LPIPS_list)
  plt.xlabel('Iteration/5')
  plt.title('LPIPS')
      
  plt.savefig(path_0+'/avg_metrics.png')

  torch.save(avg_PSNR_list, path_0 + '/avg_PSNR_list.pt')
  torch.save(avg_SSIM_list, path_0 + '/avg_SSIM_list.pt')
  torch.save(avg_LPIPS_list, path_0 + '/avg_LPIPS_list.pt')    


if __name__ == '__main__':
  torch.manual_seed(0)
  np.random.seed(0)
  main()
