ALPHA=3.2
EXPERT_SIZE=large
MODEL_DIR=/home/chiyu94/scratch/hashtag_paraphrase/formality/fine-tuned
PROMPTS_DATASET=/home/chiyu94/scratch/dexperts/prompts/sentiment_prompts-10k/neutral_prompts.jsonl
OUTPUT_DIR=/home/chiyu94/scratch/hashtag_paraphrase/formality/generate_out/formal/

python -m /home/chiyu94/scratch/hashtag_paraphrase/DExperts/scripts/run_sentiment_experiment.py \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type dexperts \
    --model gpt2-large \
    --pos-model $MODEL_DIR/formal \
    --neg-model $MODEL_DIR/informal \
    --alpha $ALPHA \
    --filter_p 0.9 \
    $OUTPUT_DIR
