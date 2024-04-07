#!/bin/bash
#SBATCH --job-name=demo      #作业名称
#SBATCH --partition=q_intel_gpu_nvidia_nvlink   #指定队列
#SBATCH --output=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/logs/run-job%j.out     #正常输出
#SBATCH --error=/home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/logs/run-job%j.err   #错误输出
#SBATCH --gres=gpu:1       #指定GPU数量
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --nodelist=gpu004  #指定GPU节点

module load amd/openmpi/4.0.4/gcc10.2.0 
export PATH=/home/export/base/ycsc_chenkh/hitici_02/online1/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/export/base/ycsc_chenkh/hitici_02/online1/cuda-11.8/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/export/base/ycsc_chenkh/hitici_02/online1/miniconda3/envs/evaluate/lib/python3.10/site-packages/tensorrt_libs
export LD_LIBRARY_PATH=/home/export/base/ycsc_chenkh/hitici_02/online1/cuda-11.8/lib64:$LD_LIBRARY_PATH
nvcc -V
source /home/export/base/ycsc_chenkh/hitici_02/online1/miniconda3/bin/activate

conda activate llama_factory
bash /home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/train_scripts/sft_llama.sh
# bash /home/export/base/ycsc_chenkh/hitici_02/online1/PolyLingual-LLM/LLaMA-Factory/predict_scripts/xlsum_gen.sh