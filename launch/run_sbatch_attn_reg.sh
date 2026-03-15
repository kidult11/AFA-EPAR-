#!/bin/bash
#SBATCH -A test
#SBATCH -J attn_reg
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -p short
#SBATCH -t 3-0:00:00
#SBATCH -o wetr_attn_reg.out

source /home/kidult/anaconda3/bin/activate seam
port=29501  #定义一个 Linux 环境变量port，值为 29501，这个端口是PyTorch 分布式训练的通信端口，用于单节点多进程之间的通信
crop_size=512 #图片的裁剪尺寸

file=scripts/dist_train_voc.py #指定训练的主脚本路径
config=configs/voc_attn_reg.yaml #定义环境变量config，指定训练的配置文件路径

#echo python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_final
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_final
# 移除分布式启动器，直接单进程启动
echo python $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_final --local_rank -1
python $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_final --local_rank -1