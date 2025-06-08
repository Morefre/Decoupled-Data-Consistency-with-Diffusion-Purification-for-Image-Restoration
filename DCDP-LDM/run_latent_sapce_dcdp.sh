'''
Run the code below to replicate the quantitative results presented in the paper
'''


'''
Super-resolution
'''
#  Option 1: perform diffusion purification with 20 steps DDIM
python dcdp.py --full_ddim=True --total_num_iterations=10 --csgm_num_iterations=100 --ddim_init_timestep=400 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=100000 --optimizer=SGD --lr=1000 --data_consistency_type=latent --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/super_resolution_config.yaml

#  Option 2: perform diffusion purification with 1-step Tweedie's formula. This is much faster
python dcdp.py --full_ddim=False --total_num_iterations=10 --csgm_num_iterations=100 --ddim_init_timestep=400 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=100000 --optimizer=SGD --lr=1000 --data_consistency_type=latent --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/super_resolution_config.yaml


'''
Inpainting
'''
#  Option 1: perform diffusion purification with 20 steps DDIM
python dcdp.py --full_ddim=True --total_num_iterations=20 --csgm_num_iterations=50 --ddim_init_timestep=500 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=100000 --optimizer=SGD --lr=1000 --data_consistency_type=latent --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/inpainting_config.yaml

#  Option 2: perform diffusion purification with 1-step Tweedie's formula. This is much faster
python dcdp.py --full_ddim=False --total_num_iterations=20 --csgm_num_iterations=50 --ddim_init_timestep=500 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=100000 --optimizer=SGD --lr=1000 --data_consistency_type=latent --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/inpainting_config.yaml


'''
Motion deblurring
'''
#  Option 1: perform diffusion purification with 20 steps DDIM
python dcdp.py --full_ddim=True --total_num_iterations=10 --csgm_num_iterations=100 --ddim_init_timestep=400 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=100000 --optimizer=SGD --lr=100000 --data_consistency_type=pixel --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/motion_deblur_config.yaml

#  Option 2: perform diffusion purification with 1-step Tweedie's formula. This is much faster
python dcdp.py --full_ddim=False --total_num_iterations=10 --csgm_num_iterations=100 --ddim_init_timestep=400 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=100000 --optimizer=SGD --lr=100000 --data_consistency_type=pixel --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/motion_deblur_config.yaml

'''
Gaussian deblurring
'''
#  Option 1: perform diffusion purification with 20 steps DDIM
python dcdp.py --full_ddim=True --total_num_iterations=4 --csgm_num_iterations=250 --ddim_init_timestep=400 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=100000 --optimizer=SGD --lr=100000 --data_consistency_type=pixel --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/gaussian_deblur_config.yaml

#  Option 2: perform diffusion purification with 1-step Tweedie's formula. This is much faster
python dcdp.py --full_ddim=False --total_num_iterations=4 --csgm_num_iterations=250 --ddim_init_timestep=400 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=1000000 --optimizer=SGD --lr=100000 --data_consistency_type=pixel --verbose=True --momentum=0.9 \
                --task_config=configs/tasks/gaussian_deblur_config.yaml

'''
Nonlinear deblurring
'''
#  Option 1: perform diffusion purification with 20 steps DDIM
python dcdp.py --full_ddim=True --total_num_iterations=10 --csgm_num_iterations=100 --ddim_init_timestep=1000 --ddim_end_timestep=50 \
               --save_every_main=1 --save_every_sub=100000 --optimizer=SGD --lr=5000 --data_consistency_type=latent --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/nonlinear_deblur_config.yaml

#  Option 2: perform diffusion purification with 1-step Tweedie's formula. This is much faster
python dcdp.py --full_ddim=False --total_num_iterations=10 --csgm_num_iterations=100 --ddim_init_timestep=1000 --ddim_end_timestep=50 \
               --save_every_main=1 --save_every_sub=1 --optimizer=SGD --lr=5000 --data_consistency_type=latent --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/nonlinear_deblur_config.yaml