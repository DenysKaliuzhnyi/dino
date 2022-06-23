#!/bin/bash

# The name of the job
#SBATCH -J train_dino

# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=slurm/slurm-%x.%j.out

# The job requires 1 compute node
#SBATCH -N 1

# The job requires 1 task per node
#SBATCH --ntasks-per-node=6

# The maximum walltime of the job is 5 minutes
#SBATCH -t 8-00:00:00

#SBATCH --mem=50G

#SBATCH --partition=gpu

# Indicates that you need one GPU node
#SBATCH --gres=gpu:tesla:6
#SBATCH --exclude=falcon3
# Commands to execute go below

# Load Python
module load python/3.8.6

# Activate your environment
source env/bin/activate


DATA_PATH=data_link/

# prediction of translatiom from the last checkpoint
python -m torch.distributed.launch --nproc_per_node=6 main_dino.py --arch vit_small \
                                                                   --patch_size 16 \
                                                                   --out_dim 65536 \
                                                                   --norm_last_layer true \
                                                                   --warmup_teacher_temp 0.04 \
                                                                   --teacher_temp 0.04 \
                                                                   --warmup_teacher_temp_epochs 0 \
                                                                   --use_fp16 true \
                                                                   --weight_decay 0.35 \
                                                                   --weight_decay_end 0.4 \
                                                                   --clip_grad 0 \
                                                                   --batch_size_per_gpu 128 \
                                                                   --epochs 50 \
                                                                   --freeze_last_layer 10 \
                                                                   --lr 1e-5 \
                                                                   --min_lr 1e-7 \
                                                                   --global_crops_scale 0.4 1.0 \
                                                                   --local_crops_number 8 \
                                                                   --local_crops_scale 0.05 0.4 \
                                                                   --seed 0 \
                                                                   --optimizer adamw \
                                                                   --momentum_teacher 0.9995 \
                                                                   --use_bn_in_head False \
                                                                   --drop_path_rate 0.1 \
                                                                   --saveckp_freq 1 \
                                                                   --num_workers 4 \
                                                                   --data_path "$DATA_PATH" \  # must contain directories of images per class
                                                                   --output_dir checkpoints/ \  # directory to save & pick checkpoints and loss logs
                                                                   --run_name run_01 \  # directory name to create per run inside output_dir
                                                                   --weights checkpoint_vit.pth  # checkpoint filename to start from inside output_dir
