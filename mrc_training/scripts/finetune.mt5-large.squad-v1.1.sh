
export workspace="/ist/users/lalital/.cache/"
export HF_DATASETS_CACHE="${workspace}/huggingface/datasets"
export HF_MODULES_CACHE="${workspace}/huggingface/modules/"
export XDG_CACHE_HOME="${workspace}/huggingface"

export TRANSFORMERS_CACHE="${workspace}/huggingface/transformers'"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline


# BATCH_SIZES=(8)
# LEARNING_RATES=(1.5e-5 1e-5)
EPOCHS=(1 2 3 4 5)
batch_size=16
gradient_accumulation_steps=${1}
learning_rate=${2}
# gradient_accumulation_steps = 1 , bz = 16
# gradient_accumulation_steps = 8 , bz = 128
# for batch_size in "${BATCH_SIZES[@]}"
# do
    # for learning_rate in "${LEARNING_RATES[@]}"
    # do
for epoch in "${EPOCHS[@]}"
do
    echo "batch_size: 16"
    echo "learning_rate: ${learning_rate}"
    echo "epoch: ${epoch}"

    run_name="exp001.t5-large.seq2seq.squad_hparams.bz-${batch_size}.lr-${learning_rate}.ep-${epoch}"
    echo " Run name: ${run_name}"
    mkdir -p ./logs/${run_name}/
    CUDA_VISIBLE_DEVICES=0 python3 run_seq2seq_qa.py \
    --model_name_or_path ./models/mt5-large \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --do_train \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --warmup_ratio 0.06 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gradient_accumulation_steps}
    --load_best_model_at_end True \
    --fp16 True \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${epoch} \
    --max_seq_length 512 \
    --doc_stride 128 \
    --max_answer_length 30 \
    --generation_max_length 30 \
    --output_dir ./checkpoints/${run_name} \
    --logging_dir ./logs/${run_name} |& tee -a ./logs/${run_name}/trainer.log
done