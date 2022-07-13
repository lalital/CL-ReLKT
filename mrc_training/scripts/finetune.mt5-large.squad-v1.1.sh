
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

batch_size=16
gradient_accumulation_steps=${1}
learning_rate=${2}
max_steps=${3}
save_steps=${4}
eval_steps=${5}
# gradient_accumulation_steps = 1 , bz = 16
# gradient_accumulation_steps = 8 , bz = 128

echo "batch_size:${batch_size}"
echo "learning_rate: ${learning_rate}"
echo "max_steps: ${max_steps}"
echo "save_steps: ${save_steps}"
echo "eval_steps: ${eval_steps}"

run_name="exp001.t5-large.seq2seq.squad_hparams.bz-${batch_size}.grad_acc-${gradient_accumulation_steps}.lr-${learning_rate}.max_steps-${max_steps}"
echo " Run name: ${run_name}"
mkdir -p ./logs/${run_name}/
CUDA_VISIBLE_DEVICES=0 python3 run_seq2seq_qa.py \
--model_name_or_path ./models/mt5-large \
--dataset_name squad \
--context_column context \
--question_column question \
--answer_column answers \
--do_train \
--logging_steps 10 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps ${save_steps} \
--per_device_train_batch_size ${batch_size} \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--fp16 True \
--learning_rate ${learning_rate} \
--lr_scheduler_type constant \
--optim adafactor \
--max_steps ${max_steps} \
--max_seq_length 512 \
--doc_stride 128 \
--max_answer_length 30 \
--generation_max_length 30 \
--output_dir ./checkpoints/${run_name} \
--logging_dir ./logs/${run_name} |& tee -a ./logs/${run_name}/trainer.log