# Fine-tuning for Vision to Proprioception prediction (VPP) Classification

## Description
We evaluate the usefulness of EgoPet on a robotic task, focusing on the problem of vision-based locomotion. Specifically, the task consists of predicting the parameters of the terrain a quadrupedal robot is walking on. The accurate prediction of these parameters is correlated with higher performance in the downstream walking task.

The parameters we predict are the local terrain geometry, the terrain's friction, and the parameters related to the robot's walking behavior on the terrain, including the robot's speed, motor efficiency, and high-level command. We aim to predict a latent representation $z_t$ of the terrain parameters. This latent representation consists of the hidden layer of a neural network trained in simulation to encode end-to-end together with the walking policy. 

To collect the dataset, we let the robot walk in many environments. The training dataset contains 120 thousand frames from 3 environments (approximately 2.3 hours of total walking time): an office, a park, and a beach. Each of them contains different terrain geometries, including flat, steps, and slopes. Each sample contains an image collected from a forward-looking camera mounted on the robot and the (latent) parameters of the terrain below the center of mass of the robot $z_t$ estimated with a history of proprioception.

The task consists of predicting $z_t$ from a (history) of images. We generate several sub-tasks by predicting the future terrain parameters $z_{t+0.8}, z_{t+1.5}$ and the past ones $z_{t-0.8}, z_{t-1.5}$. These time intervals were selected to differentiate between forecasting and estimation. We divide the datasets into a training and testing set per episode, i.e., two different policy runs. We construct three test datasets, one in distribution (same location and lighting conditions as training), out of distribution due to lighting conditions (same location but very different time (night)), and out of distribution data such as sand which was never experienced during training.

For more information about the VPP task refer to our paper!

## Dataset Setup
Download and extract CMS_data.tar.gz from from [here](https://drive.google.com/file/d/1ZKSWwCoZP1mHjpksIEAwh3sTeeSJNF3B/view?usp=sharing)

## Linear Probing

Run the following command to train a linear probing layer to predict the proprioception (past, present, future) given the video input.

```
DATA_PATH='./datasets/CMS/up_down_policy_data'
LOOKHEAD=0.8 # in -1.5 -0.8 0 0.8 1.5
MASTER_PORT=29500
DATA_ROOT=${DATA_PATH}
OUTPUT_DIR="./logs_dir/mvd_vit_base_with_vit_base_teacher_egopet/finetune_on_cms_lookahead_${LOOKHEAD}_8frames_update_freq_4"
MODEL_PATH="./logs_dir/mvd_vit_base_with_vit_base_teacher_egopet/checkpoint-2669.pth"
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 --use_env \
    run_fs_domain_adaptation.py \
    --model vit_base_patch16_224 \
    --nb_classes 10 \
    --latent_dim 10 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size 256 --update_freq 4 --num_sample 4 \
    --num_frames 8 --sampling_rate 4 \
    --lr 5e-4 --epochs 50 \
    --input_use_imgs 8 \
    --lookhead ${LOOKHEAD} \
    --num_workers 5 \
    --save_ckpt_freq 20
```

### Pretrained Models
| Model             | Lookahead | Link |
|-------------------|-----------|------|
| MVD (ViT-B) | -1.5 |   [link](https://drive.google.com/file/d/12tO7LwjZ66voCTxp6lNcSaeLaU-9YlYE/view?usp=sharing)   |
| MVD (ViT-B) | -0.8 |   [link](https://drive.google.com/file/d/135ndczYtWKF04ZNl-5zs_O6yTdLh5T7O/view?usp=sharing)   |
| MVD (ViT-B) | 0 |   [link](https://drive.google.com/file/d/1r_8ZbJtuI_6ImFQFzjgIb1ERVBjq-eY-/view?usp=sharing)   |
| MVD (ViT-B) | 0.8 |   [link](https://drive.google.com/file/d/1f78c-ascoWKa3_29rq-4NhCruidMsFoI/view?usp=sharing)   |
| MVD (ViT-B) | 1.5 |   [link](https://drive.google.com/file/d/1BO5gzVs5TiNilRpF5OzuTtyVSjfMO33Z/view?usp=sharing)   |