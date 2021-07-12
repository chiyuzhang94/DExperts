#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=186G
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
PROMPTS_DATASET=/home/chiyu94/scratch/dexperts/prompts/sentiment_prompts-10k/neutral_prompts.jsonl
OUTPUT_DIR=/home/chiyu94/scratch/hashtag_paraphrase/formality/generate_out/formal/

python /home/chiyu94/scratch/hashtag_paraphrase/DExperts/scripts/run_sentiment_experiment.py \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --pos-model $MODEL_DIR/formal \
    --neg-model $MODEL_DIR/informal \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $OUTPUT_DIR
