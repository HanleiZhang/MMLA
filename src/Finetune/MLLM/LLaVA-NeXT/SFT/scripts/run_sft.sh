#!/bin/bash

## CUDA 11.8

BASE_PATH="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/zhanghanlei" 
BASE_DATA_PATH="$BASE_PATH/Datasets"
CONDA_PATH="$BASE_PATH/anaconda3/bin/conda"
WORK_PATH="$BASE_PATH/MMLA/src/Finetune/MLLM/LLaVA-NeXT/SFT"
FRAMEWORK_PATH="$BASE_PATH/MMLA/src/Frameworks/LLaVA-NeXT-main"
INFERENCE_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference/MLLM"
EVALUATION_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference"
RESULTS_PATH="$WORK_PATH/results"

export WANDB_MODE=disabled
PROMPT_VERSION="qwen_1_5"
ENV_NAME="LLaVA"

declare -A LLM_VERSION_CLEAN_map
LLM_VERSION_CLEAN_map["LLaVA-Video-7B-Qwen2"]="Qwen_Qwen2-7B-Instruct"
LLM_VERSION_CLEAN_map["LLaVA-Video-72B-Qwen2"]="Qwen_Qwen2-72B-Instruct"
LLM_VERSION_CLEAN_map["llava-onevision-qwen2-7b-ov-chat"]="Qwen_Qwen2-7B-Instruct"
LLM_VERSION_CLEAN_map["llava-onevision-qwen2-72b-ov-chat"]="Qwen_Qwen2-72B-Instruct"

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="google_siglip-so400m-patch14-384"

##################################
MODEL_NAMES=( "LLaVA-Video-72B-Qwen2" "llava-onevision-qwen2-72b-ov-chat" "llava-onevision-qwen2-7b-ov-chat" "LLaVA-Video-7B-Qwen2") 

###############################
declare -A ENV_NAMES

ENV_NAMES["LLaVA-Video-7B-Qwen2"]="LLaVA"
ENV_NAMES["LLaVA-Video-72B-Qwen2"]="LLaVA"
ENV_NAMES["llava-onevision-qwen2-7b-ov-chat"]="LLaVA"
ENV_NAMES["llava-onevision-qwen2-72b-ov-chat"]="LLaVA"

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
declare -A MODEL_DEVICES

MODEL_DEVICES["LLaVA-Video-7B-Qwen2"]="0,1,2,3,4,5,6,7" 
MODEL_DEVICES["LLaVA-Video-72B-Qwen2"]="0,1,2,3,4,5,6,7" 
MODEL_DEVICES["llava-onevision-qwen2-7b-ov-chat"]="0,1,2,3,4,5,6,7" 
MODEL_DEVICES["llava-onevision-qwen2-72b-ov-chat"]="0,1,2,3,4,5,6,7"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    BASE_MODEL_PATH="$BASE_PATH/llm_checkpoints/$MODEL_NAME"
    DEVICE_IDS="${MODEL_DEVICES[$MODEL_NAME]}"

    for dataset in "${!task_map[@]}"; do
        DATASET_NAME=$dataset
        TASK_NAME="${task_map[$dataset]}"
        ENV_NAME="${ENV_NAMES[$MODEL_NAME]}"

        timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

        ##################################### Path and Parameter Configuration #####################################
        RUN_NAME="${MODEL_NAME}-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN_map[$MODEL_NAME]}-${DATASET}-${TASK}"
        DATA_JSON="llava-next"
        MODEL_PATH="$BASE_PATH/models_sft/$DATASET_NAME/$TASK_NAME/${MODEL_NAME}_SFT"
        JSON_DATA_PATH="$FRAMEWORK_PATH/data/${DATASET_NAME}_train_${TASK_NAME}_${DATA_JSON}.json"
        VIDEO_FOLDER="$BASE_DATA_PATH/$DATASET_NAME/video"
        IMAGE_FOLDER="$VIDEO_FOLDER"
        LORA_PATH=$MODEL_PATH/$RUN_NAME
        
        json_file="$WORK_PATH/configs/${MODEL_NAME}.json"
        echo "JSON file: $json_file"
        
        dataset_section=$(jq -r --arg dataset_name "$DATASET_NAME" '.[$dataset_name]' "$json_file")
        echo "Dataset section: $dataset_section"
        learning_rate=$(echo "$dataset_section" | jq -r '.learning_rate')
        num_train_epochs=$(echo "$dataset_section" | jq -r '.num_train_epochs')
        warmup_ratio=$(echo "$dataset_section" | jq -r '.warmup_ratio')
        train_batch_size=$(echo "$dataset_section" | jq -r '.train_batch_size')
        lora_r=$(echo "$dataset_section" | jq -r '.lora_r')
        lora_alpha=$(echo "$dataset_section" | jq -r '.lora_alpha')
        gradient_accumulation_steps=$(echo "$dataset_section" | jq -r '.gradient_accumulation_steps')
        model_max_length=$(echo "$dataset_section" | jq -r '.model_max_length')
        dataloader_num_workers=$(echo "$dataset_section" | jq -r '.dataloader_num_workers')
        deepspeed=$(echo "$dataset_section" | jq -r '.deepspeed')
        weight_decay=$(echo "$dataset_section" | jq -r '.weight_decay')
        save_steps=$(echo "$dataset_section" | jq -r '.save_steps')
        save_total_limit=$(echo "$dataset_section" | jq -r '.save_total_limit')
    
        echo "Learning rate: $learning_rate"

        ##################################### TRAINING #####################################
        echo "[$(date)] Start training: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"
        $CONDA_PATH run -n ${ENV_NAME} --no-capture-output deepspeed $FRAMEWORK_PATH/llava/train/train_mem.py \
            --deepspeed $FRAMEWORK_PATH/scripts/${deepspeed} \
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
            --num_train_epochs $num_train_epochs \
            --lora_r $lora_r \
            --lora_alpha $lora_alpha \
            --per_device_train_batch_size $train_batch_size \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps $save_steps \
            --save_total_limit $save_total_limit \
            --learning_rate $learning_rate \
            --weight_decay $weight_decay \
            --warmup_ratio $warmup_ratio \
            --lr_scheduler_type "cosine" \
            --logging_steps 2 \
            --tf32 True \
            --model_max_length $model_max_length \
            --gradient_checkpointing True \
            --lazy_preprocess True \
            --dataloader_num_workers $dataloader_num_workers

        echo "Training has finished!"

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
done


