#!/bin/bash

## CUDA 11.8

BASE_PATH="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/zhanghanlei" 
BASE_DATA_PATH="$BASE_PATH/Datasets"
CONDA_PATH="$BASE_PATH/anaconda3/bin/conda"
WORK_PATH="$BASE_PATH/MMLA/src/Finetune/MLLM/MiniCPM-V_2-6/IT"
FRAMEWORK_PATH="$BASE_PATH/MMLA/src/Frameworks/Swift"
INFERENCE_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference/MLLM"
EVALUATION_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference"
RESULTS_PATH="$WORK_PATH/results"

MASTER_PORT=23458 
export NCCL_TIMEOUT=1201
export NCCL_BLOCKING_WAIT=1

###############################
MODEL_NAME="MiniCPM-V-2_6" 
ENV_NAME="MiniCPM"
BASE_MODEL_PATH="$BASE_PATH/llm_checkpoints/$MODEL_NAME"
DEVICE_IDS="0,1,2,3,4,5,6,7"
nproc_per_node=8

##################################### Path and Parameter Configuration #####################################

MODEL_PATH="$BASE_PATH/models_it/Instruct/${MODEL_NAME}_IT"

echo "[$(date)] Start training: Instruction tuning MODEL: $MODEL_NAME"
CUDA_VISIBLE_DEVICES=${DEVICE_IDS} NPROC_PER_NODE=$nproc_per_node ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output swift sft \
    --model $BASE_MODEL_PATH \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --train_type lora \
    --dataset $FRAMEWORK_PATH/video_prompt_data/Instruct/train.jsonl \
    --val_dataset $FRAMEWORK_PATH/video_prompt_data/Instruct/dev.jsonl \
    --deepspeed zero2 \
    --max_length 1024 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 20 \
    --save_steps 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --output_dir "$MODEL_PATH" \

echo "Training has finished!"

##################################### Best Model Selection #####################################
sub_dir=$(find "$MODEL_PATH" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
if [ -z "$sub_dir" ]; then
    echo "No subdirectory found in $MODEL_PATH"
    continue
fi
echo "sub_dir: $sub_dir"

best_checkpoint=""
if [ -f "$sub_dir/logging.jsonl" ]; then
    echo "Found logging.jsonl in $sub_dir, extracting best_model_checkpoint..."
    best_checkpoint=$(jq -r 'select(.best_model_checkpoint != null) | .best_model_checkpoint' "$sub_dir/logging.jsonl" | head -n 1)
fi

if [ -z "$best_checkpoint" ]; then
    echo "No best_model_checkpoint found in logging.jsonl, falling back to checkpoint search..."
    max_checkpoint=$(find "$sub_dir" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | \
                        awk -F'checkpoint-' '{print $2}' | sort -nr | head -n1)
    if [ -z "$max_checkpoint" ]; then
        echo "No checkpoint found in $sub_dir"
        continue
    fi
    echo "max_checkpoint: $max_checkpoint"
    best_checkpoint="$sub_dir/checkpoint-${max_checkpoint}"
fi

adapters_path="$best_checkpoint"
echo "Using adapter: $adapters_path"

##################################
declare -A task_map
task_map['IEMOCAP']='emotion'
task_map['AnnoMi-client']='communication_behavior'
task_map['MELD']='emotion'
task_map['MELD-DA']='dialogue_act'
task_map['MOSI']='sentiment'
task_map['MUStARD']='speaking_style'
task_map['MIntRec']='intent'
task_map['IEMOCAP-DA']='dialogue_act'
task_map['MIntRec2.0']='intent'
task_map['Ch-sims']='sentiment'
task_map['UR-FUNNY']='speaking_style'
task_map['AnnoMi-therapist']='communication_behavior'

for dataset in "${!task_map[@]}"; do
    DATASET_NAME=$dataset
    TASK_NAME="${task_map[$dataset]}"
    timestamp=$(date +"%Y-%m-%d-%H-%M-%S")


    ##################################### INFERENCE #####################################

    echo "[$(date)] Start inference: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"

    CUDA_VISIBLE_DEVICES=${DEVICE_IDS} NPROC_PER_NODE=$nproc_per_node ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output swift infer \
        --adapters $adapters_path \
        --infer_backend pt \
        --val_dataset $FRAMEWORK_PATH/video_prompt_data/$DATASET_NAME/test.jsonl \
        --result_path $RESULTS_PATH/${DATASET_NAME}_${TASK_NAME}_result.jsonl

    echo "Inference finished!"

    ##################################### EVALUATION #####################################

    echo "[$(date)] Start evaluation: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"

    ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output python ${EVALUATION_PATH}/eval.py \
        --dataset $DATASET_NAME \
        --task $TASK_NAME \
        --model $MODEL_NAME \
        --results_path $RESULTS_PATH \
        --timestamp $timestamp 

    echo "Evaluation has finished!"
done

