#!/bin/bash

## CUDA 11.8

BASE_PATH="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/zhanghanlei" 
BASE_DATA_PATH="$BASE_PATH/Datasets"
CONDA_PATH="$BASE_PATH/anaconda3/bin/conda"
WORK_PATH="$BASE_PATH/MMLA/src/Finetune/MLLM/VideoLLaMA2/SFT"
FRAMEWORK_PATH="$BASE_PATH/MMLA/src/Frameworks/VideoLLaMA2"
INFERENCE_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference/MLLM"
EVALUATION_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference"
RESULTS_PATH="$WORK_PATH/results"

cd $FRAMEWORK_PATH

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16670
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=1
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=videollama2
export TRITON_CACHE_DIR=${BASE_PATH}/
export DECORD_EOF_RETRY_MAX=20480

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

###############################
MODEL_NAME="VideoLLaMA2-7B-Base" 
ENV_NAME="VideoLLaMA2"
BASE_MODEL_PATH="$BASE_PATH/llm_checkpoints/$MODEL_NAME"
DEVICE_IDS="0,1,2,3,4,5,6,7"

for dataset in "${!task_map[@]}"; do
    DATASET_NAME=$dataset
    TASK_NAME="${task_map[$dataset]}"
    
    timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

    ##################################### Path and Parameter Configuration #####################################
    RUN_NAME="${MODEL_NAME}-${DATASET_NAME}-${TASK_NAME}-lora"
    MODEL_PATH="$BASE_PATH/models_sft/$DATASET_NAME/$TASK_NAME/${MODEL_NAME}_SFT"
    
    echo "[$(date)] Start training: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"
    $CONDA_PATH run -n ${ENV_NAME} --no-capture-output torchrun --nnodes $WORLD_SIZE \
        --nproc_per_node $NPROC_PER_NODE \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --node_rank $RANK \
        videollama2/train_flash_attn.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --deepspeed scripts/zero2.json \
        --model_type videollama2 \
        --model_path ${BASE_PATH}/llm_checkpoints/Mistral-7B-Instruct-v0.2 \
        --vision_tower ${BASE_PATH}/llm_checkpoints/openaiclip-vit-large-patch14-336 \
        --mm_projector_type stc_connector \
        --pretrain_mm_mlp_adapter ${BASE_PATH}/llm_checkpoints/${MODEL_NAME}/mm_projector.bin \
        --data_path   ${FRAMEWORK_PATH}/data/${DATASET_NAME}_train_${TASK_NAME}.json \
        --data_folder ${BASE_DATA_PATH}/$DATASET_NAME/video \
        --mm_vision_select_layer -2 \
        --image_aspect_ratio pad \
        --num_frames 8 \
        --bf16 False \
        --tf32 False \
        --fp16 True \
        --output_dir $MODEL_PATH/${RUN_NAME} \
        --num_train_epochs 10 \
        --per_device_train_batch_size $LOCAL_BATCH_SIZE \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 99 \
        --learning_rate 2e-5 \
        --weight_decay 0.1 \
        --warmup_ratio 0.5 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --report_to tensorboard \
        --run_name "$RUN_NAME"

    echo "Training has finished!"

    ##################################### INFERENCE #####################################

    echo "[$(date)] Start inference: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"

    ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output python ${INFERENCE_PATH}/VideoLLaMA2/infer.py \
        --base_model_path $BASE_MODEL_PATH \
        --base_data_path $BASE_DATA_PATH \
        --dataset $DATASET_NAME \
        --results_path $RESULTS_PATH/$MODEL_NAME \
        --task $TASK_NAME \
        --device_ids $DEVICE_IDS \

    echo "Inference finished!"

    ##################################### EVALUATION #####################################

    echo "[$(date)] Start evaluation: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"
    
    ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output python ${EVALUATION_PATH}/eval.py \
        --dataset $DATASET_NAME \
        --task $TASK_NAME \
        --model $MODEL_NAME \
        --results_path $RESULTS_PATH/$MODEL_NAME \
        --timestamp $timestamp 

    echo "Evaluation has finished!"
done


