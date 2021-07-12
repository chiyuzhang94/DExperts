#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=generate
#SBATCH --output=./out/generate.out
#SBATCH --account=rrg-mageed
#SBATCH --mail-user=zcy94@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

module load python/3.8
module load gcc arrow 

module load cuda cudnn
module load openmpi nccl
module load python scipy-stack
#module load openblas
source ~/py38_tf46/bin/activate
#source ~/.bashrc
#conda activate py36_tf46

python3 generation_t5.py \
	--input_file /home/chiyu94/scratch/hashtag_paraphrase/transfer_data_binary/sad/val.tsv \
	--top_k 0 \
	--top_p 0.9 \
	--output_dir "/home/chiyu94/scratch/hashtag_paraphrase/evaluation/generation_out" \
	--batch_size 8 \
	--base_model_name ../distributed/st5-para_binary/ \
    --expert_model_name ../distributed/st5_mul_joy/checkpoint-93330/ \
    --anti_model_name ../distributed/st5_mul_sad/checkpoint-101983/ \
    --source_cls "sad" \
    --target_cls "joy"
    
python3 generation_t5.py \
	--input_file /home/chiyu94/scratch/hashtag_paraphrase/transfer_data_binary/joy/val.tsv \
	--top_k 0 \
	--top_p 0.9 \
	--output_dir "/home/chiyu94/scratch/hashtag_paraphrase/evaluation/generation_out" \
	--batch_size 8 \
	--base_model_name ../distributed/st5-para_binary/ \
    --expert_model_name ../distributed/st5_mul_sad/checkpoint-101983/ \
    --anti_model_name ../distributed/st5_mul_joy/checkpoint-93330/ \
    --source_cls "joy" \
    --target_cls "sad"
