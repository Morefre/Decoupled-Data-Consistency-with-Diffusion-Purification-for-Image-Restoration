from ldm_inverse.condition_methods import get_conditioning_method
from ldm.models.diffusion.ddim import DDIMSampler
from data.dataloader import get_dataset, get_dataloader
from scripts.utils import clear_color, mask_generator
import matplotlib.pyplot as plt
from ldm_inverse.measurements import get_noise, get_operator
from functools import partial
import numpy as np
import yaml
from model_loader import load_model_from_config, load_yaml
import os
import torch
import torchvision.transforms as transforms
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compare_ssim # latest scikit-image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Learned Perceptual Image Patch Similarity
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

def normalize_image(img):
  img = torch.clamp(img, min=-1, max=1)
  img = (img+1)/2
  return img

def normalize_code(code):
  code = code-torch.min(code)
  code = code/torch.max(code)
  return code

def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)
    return model

def CSGM_Solver_Latent_Space(model, measurements, z_init, num_iterations, device, operator, mask=None, 
                             optimizer = 'Adam', momentum=0, lr=0.1, save_every=50):
    '''
    img_gt: torch tensor with size (b,c,h,w)
    z_init: initial latent code
    mask: inpainting mask
    '''
    if mask != None:
      mask = mask.to(device)

    model.to(device)
    model.eval()

    z = z_init.clone().detach().requires_grad_(True)

    if optimizer == 'Adam':
       optimizer = torch.optim.Adam([z],lr=lr)

    elif optimizer == 'SGD':
      optimizer = torch.optim.SGD([z],lr=lr,momentum=momentum)

    criterion = torch.nn.MSELoss().to(device)

    measurements = measurements.clone().detach()

    z_list = []     # a list for storing intermediate reconstructions

    for i in range(num_iterations):
      optimizer.zero_grad()
      recon = model.differentiable_decode_first_stage(z)
      if mask != None:
         recon_measurements = operator.forward(recon,mask=mask)
         recon_loss = criterion(measurements, recon_measurements)
      else:
         recon_measurements = operator.forward(recon)
         recon_loss = criterion(measurements, operator.forward(recon))

      recon_loss.backward()
      optimizer.step()

      if i % save_every == 0 or i == num_iterations-1: 
         z_list.append(torch.tensor(z))

    return z, z_list

def CSGM_Solver_Pixel_Space(model, measurements, x_init, num_iterations, device, operator, mask=None, 
                            optimizer = 'Adam', momentum=0.9, lr=0.1, save_every=50):
    '''
    img_gt: torch tensor with size (b,c,h,w)
    z_init: initial latent code
    mask: inpainting mask
    '''
    if mask != None:
       mask = mask.to(device)
  
    model.to(device)
    model.eval()

    x = x_init.clone().detach().requires_grad_(True)
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam([x],lr=lr)

    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD([x],lr=lr,momentum=momentum)
    
    criterion = torch.nn.MSELoss().to(device)

    measurements = measurements.clone().detach()

    z_list = []         # a list for storing intermediate reconstructions

    for i in range(num_iterations):
        optimizer.zero_grad()
        recon = x
        if mask != None:
           recon_loss = criterion(measurements, operator.forward(recon,mask=mask))
        else:
           recon_loss = criterion(measurements, operator.forward(recon))

        recon = normalize_image(recon)
        recon_loss.backward()
        optimizer.step()

        if i % save_every == 0 or i == num_iterations-1:
           with torch.no_grad():
              with model.ema_scope():
                z = model.encode_first_stage(torch.tensor(x).to(device))
                z = model.get_first_stage_encoding(z)
                z_list.append(torch.tensor(z))

    return z_list[-1], z_list


def Diffusion_Purified_CSGM(model, img_gt, z_dim, total_num_iterations, csgm_num_iterations, device,
                            ddim_init_timestep, ddim_end_timestep, operator, noise_std, inverse_problem_type, data_consistency_type='pixel',
                            mask=None, full_ddim = True, ddim_eta=0, ddim_num_iterations=20, purification_schedule='constant', 
                            optimizer='Adam', momentum=0, lr=0.1, save_every_main=50, 
                            save_every_sub=1, verbose=False, root_path=None):
   
   DDIM_Sampler = DDIMSampler(model)

   if mask != None:
      mask = mask.to(device)

   img_gt = img_gt.to(device)
   model.to(device)
   model.eval()

   if inverse_problem_type == 'nonlinear_blur':
      np.random.seed(0) # Comment this line if use a random kernel for each image this might need to be np.random
      kernel_np = np.random.randn(1,512,2,2)*1.2
      random_kernel = (torch.from_numpy(kernel_np)).float().to(device)
      #random_kernel = torch.randn(1, 512, 2, 2).to(device) * 1.2
      operator.random_kernel = random_kernel
   
   if mask != None:
      measurements = operator.forward(img_gt,mask=mask)
   else:
      measurements = operator.forward(img_gt)
   
   measurements = measurements + torch.randn(measurements.shape).to(device)*noise_std

   plt.figure()
   plt.imshow(torch_to_np(normalize_image(measurements)))
   plt.title('Measurements')
   plt.savefig(root_path+'measurements.png')

   # initialize x as 0 if enforcing data consistency in the pixel space
   if data_consistency_type == 'pixel':
      x = torch.zeros(img_gt.shape, device=device, requires_grad=True)
   
   # initialize latent code z as a random vectror if enforcing data consistency in the latent space
   elif data_consistency_type == 'latent':
      z = torch.randn(z_dim, device=device, requires_grad=True)

   z_list_complete = []

   purification_timesteps = Purification_Schedule(total_num_iterations,ddim_init_timestep,ddim_end_timestep,schedule_type=purification_schedule)
   print(purification_timesteps)
   
   for i in range(total_num_iterations):
      ddim_timestep = purification_timesteps[i]
      
      # Phase 1, enforcing data consistency
      if data_consistency_type == 'latent':
         z, z_list_sub = CSGM_Solver_Latent_Space(model, measurements, z, csgm_num_iterations, device, operator=operator, mask=mask, 
                                                  optimizer=optimizer, momentum=momentum, lr=lr, save_every=save_every_sub)
      
      elif data_consistency_type == 'pixel':
         z, z_list_sub = CSGM_Solver_Pixel_Space(model, measurements, x, csgm_num_iterations, device, operator=operator, mask=mask, 
                                                 optimizer=optimizer, momentum=momentum, lr=lr, save_every=save_every_sub)

      # Phase 2: Purify the current z
      with torch.no_grad():
         with model.ema_scope():
            if data_consistency_type == 'latent':
               # Re-encode
               z = model.encode_first_stage(model.decode_first_stage(z))
               z = model.get_first_stage_encoding(z)
            
            # Prepare condition label
            xc = torch.tensor([1000])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

            z_noisy = model.q_sample(z, torch.tensor(ddim_timestep-1).unsqueeze(0).to(device))
            
            if full_ddim == True:
               if ddim_timestep <= ddim_num_iterations:
                  ddim_num_iters = ddim_timestep
               else:
                  ddim_num_iters = ddim_num_iterations
               # run the full DDIM reverse process (DCDP-LDM-I)
               z_purified, _ = DDIM_Sampler.sample(S=ddim_num_iters, batch_size=z.shape[0], shape=z.shape[1:], conditioning=c, verbose=False, x_T=z_noisy, eta=ddim_eta, custome=True, initial_timestep=ddim_timestep)
            else:
               # Predict z_0 from z_t using Tweedie's formula (DCDP-LDM-II)
               predicted_noise = model.apply_model(z_noisy,torch.tensor(ddim_timestep-1).unsqueeze(0).to(device),cond=c)
               z_purified = model.predict_start_from_noise(z_noisy,torch.tensor(ddim_timestep-1).unsqueeze(0).to(device),predicted_noise)

      z_prev = z.clone().detach()
      z = z_purified
      z_list_complete = z_list_complete + z_list_sub
      z_list_complete.append(z)
      
      # Update x for next round of data consistency optimization
      if data_consistency_type == 'pixel':
         with model.ema_scope():
            x = model.decode_first_stage(z)    

      # Save qualitative results
      if i % save_every_main == 0 or i==total_num_iterations-1:
         if verbose == True:
            plt.figure(figsize=(40,10))
            plt.subplot(141)
            plt.title('gt z')
            plt.imshow(torch_to_np(normalize_image(img_gt)))
            plt.subplot(142)
            plt.title('z')
            plt.imshow(torch_to_np(normalize_image(model.decode_first_stage(z_prev))))
            plt.subplot(143)
            plt.title('z_noisy')
            plt.imshow(torch_to_np(normalize_image(model.decode_first_stage(z_noisy))))
            plt.subplot(144)
            plt.title('z_purified')
            plt.imshow(torch_to_np(normalize_image(model.decode_first_stage(z_purified))))
            fig_name = 'Iter_'+str(i)+'.png'
            plt.savefig(root_path+fig_name)

   return z, z_list_complete

def Purification_Schedule(num_purification_steps, initial_timestep, end_timestep=0, schedule_type='linear'):
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

def load_yaml(file_path: str) -> dict:
   with open(file_path) as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
   return config

if __name__ == '__main__':
   torch.manual_seed(0)
   np.random.seed(0)
   device = torch.device('cuda')

   parser = argparse.ArgumentParser()
   
   parser.add_argument('--save_dir', type=str, default='./purification_results_imagenet')

   # Model configuration
   parser.add_argument('--ldm_config', default="configs/latent-diffusion/cin256-v2.yaml", type=str)
   parser.add_argument('--diffusion_config', default="models/ldm_imagenet/model.ckpt", type=str)
   parser.add_argument('--task_config', default="configs/tasks/nonlinear_deblur_imagenet_config.yaml", type=str)
   parser.add_argument('--z_dim', default=(1,3,64,64))

   # DCDP configuration
   parser.add_argument('--full_ddim', default=False, type=str)         # Set to False if using Tweedie's formulat for diffusion purification, set to True if using reverse ode (DDIM) for diffusion purificaiton
   parser.add_argument('--total_num_iterations', default=10, type=int)  # Total number of iterations K
   parser.add_argument('--csgm_num_iterations', default=100, type=int)  # Number of graident steps tau for each round of data consistency optimizaiton
   parser.add_argument('--ddim_init_timestep', default=400, type=int)   # initial noise level for diffusion purification
   parser.add_argument('--ddim_end_timestep', default=0, type=int)      # Final noise level for diffusion purification
   parser.add_argument('--purification_schedule', default='linear', type=str)  # time schedule for diffusion purification
   parser.add_argument('--ddim_eta', default=0.0, type=float)           # eta=0 corresponds to reverse ODE
   parser.add_argument('--ddim_num_iterations', default=20, type=int)   # number of reverse ODE steps for diffusion purification

   parser.add_argument('--save_every_main', default=1, type=int)   # initial noise level for diffusion purification
   parser.add_argument('--save_every_sub', default=10000, type=int)      # Final noise level for diffusion purification

   parser.add_argument('--optimizer', default='SGD', type=str)     
   parser.add_argument('--lr', default=1e3, type=float)          
   parser.add_argument('--momentum', default=0.9, type=float)         
   parser.add_argument('--data_consistency_type', default='pixel', type=str)         

   parser.add_argument('--verbose', default=True, type=str)       


   args = parser.parse_args()
   
   # Load configuration
   task_config = load_yaml(args.task_config)

   # Load model
   model = get_model(args)

   # Prepare dataloader
   data_config = task_config['data']
   transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )
   dataset = get_dataset(**data_config, transforms=transform)
   loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)   

   # Set up forward operator
   mask = None
   measure_config = task_config['measurement']
   operator = get_operator(device=device, **measure_config['operator'])
   print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

   if measure_config['operator']['name'] == 'inpainting':
      mask = torch.ones(1,3,256,256)
      mask[:,:,100:200,100:200] = 0

   inverse_problem_type = measure_config['operator']['name']

   noise_std = measure_config['noise']['sigma']
   z_dim = args.z_dim
   full_ddim = args.full_ddim
   if full_ddim == 'True':
      full_ddim = True
   else:
      full_ddim = False
   total_num_iterations = args.total_num_iterations
   csgm_num_iterations = args.csgm_num_iterations
   ddim_init_timestep = args.ddim_init_timestep
   ddim_end_timestep = args.ddim_end_timestep
   purification_schedule = args.purification_schedule
   ddim_eta = args.ddim_eta
   ddim_num_iterations = args.ddim_num_iterations
   save_every_main = args.save_every_main
   save_every_sub = args.save_every_sub

   optimizer = args.optimizer
   lr = args.lr
   momentum = args.momentum
   data_consistency_type = args.data_consistency_type

   verbose = args.verbose
   if verbose == 'True':
      verbose = True
   else:
      verbose = False


   # Working directory
   out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
   os.makedirs(out_path, exist_ok=True)
   
   # path for data saving
   print(full_ddim)
   path_0 = os.path.join(out_path, 'noise_std_'+str(noise_std), str(ddim_init_timestep)+'_'+str(ddim_end_timestep)+'_'+str(total_num_iterations)+'_'+str(csgm_num_iterations)+'_'+purification_schedule+'_'+str(lr)+'_'+str(momentum)+'_'+str(full_ddim)+'_'+str(ddim_num_iterations)+'_'+data_consistency_type)

   PSNR_list_All = []
   SSIM_list_All = []
   LPIPS_list_All = []
   
   for i, ref_img in enumerate(loader):
      #print(ref_img.shape)
      ref_img = torch.tensor(ref_img).to(device)
      #print(ref_img.shape)

      root_path = path_0 + '/img_' + str(i) + '/'
      if not os.path.exists(root_path):
         os.makedirs(root_path)
      
      figure_root_path = root_path + 'figures/'
      if not os.path.exists(figure_root_path):
         os.makedirs(figure_root_path)
   
      # Run DCDP
      z, z_list_complete = Diffusion_Purified_CSGM(model, img_gt=ref_img, z_dim=z_dim, total_num_iterations=total_num_iterations, csgm_num_iterations=csgm_num_iterations,
                                                   device=device, ddim_init_timestep=ddim_init_timestep, ddim_end_timestep=ddim_end_timestep, operator=operator, noise_std=noise_std,
                                                   data_consistency_type=data_consistency_type, mask=mask, full_ddim=full_ddim, ddim_eta=ddim_eta, ddim_num_iterations=ddim_num_iterations,
                                                   purification_schedule=purification_schedule, optimizer=optimizer, momentum=momentum, lr=lr, save_every_main=save_every_main, save_every_sub=save_every_sub,
                                                   verbose=verbose, root_path=figure_root_path, inverse_problem_type=inverse_problem_type)
      
      torch.save(z_list_complete, os.path.join(root_path,'z_list_complete.pt'))
      img_np = torch_to_np(ref_img)
      img_np = np.clip(img_np,-1,1)
      img_np = (img_np+1)/2
      PSNR_list = []
      SSIM_list = []
      LPIPS_list = []

      for j in range(len(z_list_complete)):
        z = z_list_complete[j]
        recon = model.decode_first_stage(z).detach().cpu()
        recon_np = recon[0].permute(1,2,0).numpy()
        recon_np = np.clip(recon_np,-1,1)
        recon_np = (recon_np+1)/2
        PSNR_list.append(peak_signal_noise_ratio(img_np,recon_np))
        SSIM_list.append(compare_ssim(img_np, recon_np, channel_axis=2, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        LPIPS_list.append(get_lpips(ref_img,recon,lpips=lpips,device=device))
      # Visualize PSNR, SSIM and LPIPS
      plt.figure(figsize=(30,10))
      plt.subplot(131)
      plt.plot(PSNR_list)
      plt.xlabel('Iteration')
      plt.title('CSGM Results for '+ inverse_problem_type+' (PSNR)')

      plt.subplot(132)
      plt.plot(SSIM_list)
      plt.xlabel('Iteration')
      plt.title('CSGM Results for '+ inverse_problem_type+' (SSIM)')

      plt.subplot(133)
      plt.plot(LPIPS_list)
      plt.xlabel('Iteration')
      plt.title('CSGM Results for '+ inverse_problem_type+' (LPIPS)')

      plt.savefig(figure_root_path+'metrics.png')
      
      print('When the measurement has additional noise, the purification in the last iteration can improve the final reconstruction quality. Otherwise, applying purification in the last iteration can degrade reconstruction quality.')
      print('Final PSNR before Purification: ',PSNR_list[-2],'Final SSIM before Purification : ',SSIM_list[-2],'Final LPIPS before Purification: ',LPIPS_list[-2])
      print('Final PSNR after Purification: ',PSNR_list[-1],'Final SSIM after Purification : ',SSIM_list[-1],'Final LPIPS after Purification: ',LPIPS_list[-1])

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
   avg_SSIM_list = np.mean(SSIM_list_All, axis=0)
   avg_LPIPS_list = np.mean(LPIPS_list_All, axis=0)

   std_PSNR_list = np.std(PSNR_list_All, axis=0)
   std_SSIM_list = np.std(SSIM_list_All, axis=0)
   std_LPIPS_list = np.mean(LPIPS_list_All, axis=0)
   
   print('When the measurement has additional noise, the purification in the last iteration can improve the final reconstruction quality. Otherwise, applying purification in the last iteration can degrade reconstruction quality.')

   print('Final Metrics before Purification:')
   print('Final average PSNR: ',avg_PSNR_list[-2],'Final average SSIM: ', avg_SSIM_list[-2],'Final average LPIPS: ',avg_LPIPS_list[-2])
   print('Final std PSNR: ',std_PSNR_list[-2],'Final std SSIM: ', std_SSIM_list[-2],'Final std LPIPS: ',std_LPIPS_list[-2])

   print('Final Metrics after Purification:')
   print('Final average PSNR: ',avg_PSNR_list[-1],'Final average SSIM: ', avg_SSIM_list[-1],'Final average LPIPS: ',avg_LPIPS_list[-1])
   print('Final std PSNR: ',std_PSNR_list[-1],'Final std SSIM: ', std_SSIM_list[-1],'Final std LPIPS: ',std_LPIPS_list[-1])

   plt.figure(figsize=(30,10))
   plt.subplot(131)
   plt.plot(avg_PSNR_list)
   plt.xlabel('Iteration')
   plt.title('Results for Large Region Inpainting (PSNR)')

   plt.subplot(132)
   plt.plot(avg_SSIM_list)
   plt.xlabel('Iteration')
   plt.title('Results for Large Region Inpainting (SSIM)')

   plt.subplot(133)
   plt.plot(avg_LPIPS_list)
   plt.xlabel('Iteration/5')
   plt.title('Results for Large Region Inpainting (LPIPS)')
      
   plt.savefig(path_0+'/avg_metrics.png')

   torch.save(avg_PSNR_list, path_0 + '/avg_PSNR_list.pt')
   torch.save(avg_SSIM_list, path_0 + '/avg_SSIM_list.pt')
   torch.save(avg_LPIPS_list, path_0 + '/avg_LPIPS_list.pt')

    