#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=dexpert_for
#SBATCH --output=dexpert_for.out
#SBATCH --account=rrg-mageed
#SBATCH --mail-user=zcy94@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

module load python/3.8
module load gcc arrow

module load cuda cudnn
module load openmpi nccl
module load python scipy-stack
source ~/py38_tf46/bin/activate

ALPHA=3.2
EXPERT_SIZE=large
MODEL_DIR=/home/chiyu94/scratch/hashtag_paraphrase/formality/fine-tuned
PROMPTS_DATASET=/home/chiyu94/scratch/dexperts/prompts/sentiment_prompts-10k/neutral_prompts2.jsonl
OUTPUT_DIR=/home/chiyu94/scratch/hashtag_paraphrase/formality/generate_out/formal/

python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model /home/chiyu94/scratch/hashtag_paraphrase/gpt2-large \
    --pos-model $MODEL_DIR/formal/checkpoint-9000 \
    --neg-model $MODEL_DIR/informal/checkpoint-9000 \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $OUTPUT_DIR
