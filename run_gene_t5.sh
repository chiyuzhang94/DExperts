#!/bin/bash
#SBATCH --time=12:00:00
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
	--input_file /home/chiyu94/scratch/hashtag_paraphrase/formality/invers_para_data/infor/val.tsv \
	--top_k 0 \
	--top_p 0.9 \
    --alpha 3.2 \
	--output_dir "/home/chiyu94/scratch/hashtag_paraphrase/formality/generate_out" \
	--batch_size 8 \
	--base_model_name /home/chiyu94/scratch/hashtag_paraphrase/Paraphrase-with-T5 \
    --expert_model_name /home/chiyu94/scratch/hashtag_paraphrase/formality/ckpt/t5_mul_for \
    --anti_model_name /home/chiyu94/scratch/hashtag_paraphrase/formality/ckpt/t5_mul_infor \
    --source_cls "informal" \
    --target_cls "formal"
    
python3 generation_t5.py \
	--input_file /home/chiyu94/scratch/hashtag_paraphrase/formality/invers_para_data/for/val.tsv \
	--top_k 0 \
	--top_p 0.9 \
    --alpha -3.2 \
	--output_dir "/home/chiyu94/scratch/hashtag_paraphrase/formality/generate_out" \
	--batch_size 8 \
	--base_model_name /home/chiyu94/scratch/hashtag_paraphrase/Paraphrase-with-T5 \
    --expert_model_name /home/chiyu94/scratch/hashtag_paraphrase/formality/ckpt/t5_mul_for \
    --anti_model_name /home/chiyu94/scratch/hashtag_paraphrase/formality/ckpt/t5_mul_infor \
    --source_cls "formal" \
    --target_cls "informal"
