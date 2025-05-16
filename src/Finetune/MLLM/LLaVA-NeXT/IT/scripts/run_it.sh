#!/bin/bash

## CUDA 11.8

BASE_PATH="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/zhanghanlei" 
BASE_DATA_PATH="$BASE_PATH/Datasets"
CONDA_PATH="$BASE_PATH/anaconda3/bin/conda"
WORK_PATH="$BASE_PATH/MMLA/src/Finetune/MLLM/LLaVA-NeXT/IT"
FRAMEWORK_PATH="$BASE_PATH/MMLA/src/Frameworks/LLaVA-NeXT-main"
INFERENCE_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference/MLLM"
EVALUATION_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference"
RESULTS_PATH="$WORK_PATH/results"

export WANDB_MODE=disabled
PROMPT_VERSION="qwen_1_5"
ENV_NAME="LLaVA"

LLM_VERSION_CLEAN="Qwen_Qwen2-72B-Instruct"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="google_siglip-so400m-patch14-384"

###############################
MODEL_NAME="LLaVA-Video-72B-Qwen2" 
DEVICE_IDS="0,1,2,3,4,5,6,7" 
ENV_NAME="LLaVA"
RUN_NAME="${MODEL_NAME}-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}"

DATA_JSON="llava-next"
VIDEO_FOLDER="/"
IMAGE_FOLDER="$VIDEO_FOLDER"
BASE_MODEL_PATH="$BASE_PATH/llm_checkpoints/$MODEL_NAME"
MODEL_PATH="$BASE_PATH/models_it/Instruct/${MODEL_NAME}_IT"
LORA_PATH=$MODEL_PATH/$RUN_NAME

JSON_DATA_PATH="$FRAMEWORK_PATH/data/Instruct_train_${DATA_JSON}.json"

##################################### TRAINING #####################################

echo "[$(date)] Start training: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"

$CONDA_PATH run -n ${ENV_NAME} --no-capture-output deepspeed $FRAMEWORK_PATH/llava/train/train_mem.py \
    --deepspeed $FRAMEWORK_PATH/scripts/zero3_offload.json \
    --lora_enable True \
    --model_name_or_path $BASE_MODEL_PATH \
    --version $PROMPT_VERSION \
    --data_path $JSON_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --vision_tower $VISION_MODEL_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir $MODEL_PATH/${RUN_NAME} \
    --num_train_epochs 3 \
    --lora_r 8 \
    --lora_alpha 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.3 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4

echo "Training has finished!"


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

    ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output python ${INFERENCE_PATH}/LLaVA-NeXT/infer.py \
        --base_model_path $BASE_MODEL_PATH \
        --base_data_path $BASE_DATA_PATH \
        --dataset $DATASET_NAME \
        --results_path $RESULTS_PATH/$MODEL_NAME \
        --task $TASK_NAME \
        --device_ids $DEVICE_IDS \
        --lora_path $LORA_PATH \
        --run_name $RUN_NAME \

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


