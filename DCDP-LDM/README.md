# DCDP: Decoupled Data Consistency via Diffusion Purification for Solving General Inverse Problems (LDM Version)


### 1) Download pretrained checkpoints (autoencoders and model)

FFHQ:
```
mkdir -p models/ldm
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./models/ldm
unzip models/ldm/ffhq.zip -d ./models/ldm

mkdir -p models/first_stage_models/vq-f4
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -P ./models/first_stage_models/vq-f4
unzip models/first_stage_models/vq-f4/vq-f4.zip -d ./models/first_stage_models/vq-f4
```

ImageNet:
```
wget https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt -P ./checkpoints/
mv checkpoints/model.ckpt models/ldm_imagenet/model.ckpt
```

<br />

### 3) Set environment

We use the external codes for motion-blurring and non-linear deblurring following the DPS codebase.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse (we already prepare this for you with necessary changes to the code for running with no error, please do not download again.)

git clone https://github.com/LeviBorodenko/motionblur motionblur
```
For nonlinear-blurring, we need to download the "GOPRO_wVAE.pth" file from [here](https://drive.google.com/file/d/1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy/view) and paste it to ./bkse/experiments/pretrained/


Install dependencies via

```
conda env create -f environment.yaml

pip install torchmetrics

pip install torchmetrics[image]
```

Follow additional instruction [here](https://github.com/CompVis/stable-diffusion/issues/72) to fix a bug in the LDM code.

<br />

### 4) Inference

```
Gaussian deblur on FFHQ

#  Option 1: perform diffusion purification with 20 steps DDIM

python dcdp.py --full_ddim=True --total_num_iterations=4 --csgm_num_iterations=250 --ddim_init_timestep=400 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=100000 --optimizer=SGD --lr=100000 --data_consistency_type=pixel --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/gaussian_deblur_config.yaml

#  Option 2: perform diffusion purification with 1-step Tweedie's formula. This is much faster

python dcdp.py --full_ddim=False --total_num_iterations=4 --csgm_num_iterations=250 --ddim_init_timestep=400 --ddim_end_timestep=0 \
               --save_every_main=1 --save_every_sub=1000000 --optimizer=SGD --lr=100000 --data_consistency_type=pixel --verbose=True --momentum=0.9 \
                --task_config=configs/tasks/gaussian_deblur_config.yaml
```


```
Nonlinear deblur on ImageNet

#  Option 1: perform diffusion purification with 20 steps DDIM

python dcdp_imagenet.py --full_ddim=True --total_num_iterations=20 --csgm_num_iterations=100 --ddim_init_timestep=300 --ddim_end_timestep=50 \
               --save_every_main=1 --save_every_sub=1000000 --optimizer=SGD --lr=5000 --data_consistency_type=latent --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/nonlinear_deblur_ImageNet_config.yaml --save_dir='./purification_results_imagenet'

#  Option 2: perform diffusion purification with 1-step Tweedie's formula. This is much faster

python dcdp_imagenet.py --full_ddim=False --total_num_iterations=20 --csgm_num_iterations=100 --ddim_init_timestep=300 --ddim_end_timestep=50 \
               --save_every_main=1 --save_every_sub=1000000 --optimizer=SGD --lr=5000 --data_consistency_type=latent --verbose=True --momentum=0.9 \
               --task_config=configs/tasks/nonlinear_deblur_ImageNet_config.yaml --save_dir='./purification_results_imagenet'
```

Please refer to [`run_latent_space_dcdp.sh`](https://github.com/Morefre/Decoupled-Data-Consistency-with-Diffusion-Purification-for-Image-Restoration/blob/main/DCDP-LDM/run_latent_sapce_dcdp.sh) for inference code of other tasks.

<br />

## Citation
If you find our work interesting, please consider citing

```
@article{li2024decoupled,
  title={Decoupled data consistency with diffusion purification for image restoration},
  author={Li, Xiang and Kwon, Soo Min and Alkhouri, Ismail R and Ravishankar, Saiprasad and Qu, Qing},
  journal={arXiv preprint arXiv:2403.06054},
  year={2024}
}
```

