#!/bin/bash

## CUDA 11.8

BASE_PATH="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/zhanghanlei" 
BASE_DATA_PATH="$BASE_PATH/Datasets"
CONDA_PATH="$BASE_PATH/anaconda3/bin/conda"
WORK_PATH="$BASE_PATH/MMLA/src/Finetune/MLLM/Qwen2-VL/IT"
FRAMEWORK_PATH="$BASE_PATH/MMLA/src/Frameworks/LLaMA-Factory-main"
INFERENCE_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference/MLLM"
EVALUATION_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference"
RESULTS_PATH="$WORK_PATH/results"

###############################
TEMPLATE="qwen2_vl"
ENV_NAME="LLaMA_Factory"
MODEL_NAMES=("Qwen2-VL-7B-Instruct" "Qwen2-VL-72B-Instruct") 

###############################
declare -A ENV_NAMES

ENV_NAMES["Qwen2-VL-72B-Instruct"]="LLaMA_Factory"
ENV_NAMES["Qwen2-VL-7B-Instruct"]="LLaMA_Factory"

###############################
declare -A MODEL_DEVICES

MODEL_DEVICES["Qwen2-VL-72B-Instruct"]="0,1,2,3,4,5,6,7"
MODEL_DEVICES["Qwen2-VL-7B-Instruct"]="0"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    BASE_MODEL_PATH="$BASE_PATH/llm_checkpoints/$MODEL_NAME"
    DEVICE_IDS="${MODEL_DEVICES[$MODEL_NAME]}"

    ENV_NAME="${ENV_NAMES[$MODEL_NAME]}"
    timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

    ##################################### Path and Parameter Configuration #####################################
    MODEL_PATH="$BASE_PATH/models_it/Instruct/${MODEL_NAME}_IT"
    MERGE_MODEL_PATH="$BASE_PATH/merge_models_it/Instruct/${MODEL_NAME}_IT"
        
    TRAIN_LORA_PATH="$WORK_PATH/train_lora"
    
    train_config="${TRAIN_LORA_PATH}/Instruct_${MODEL_NAME}.yaml"
    echo "train_config: $train_config"

    sed -i "s|model_name_or_path: .*|model_name_or_path: $BASE_MODEL_PATH|" "$train_config"
    sed -i "s|dataset_dir: .*|dataset_dir: $FRAMEWORK_PATH/data|" "$train_config"
    sed -i "s|deepspeed: .*|deepspeed: $FRAMEWORK_PATH/examples/deepspeed/ds_z3_offload_config.json|" "$train_config"
    sed -i 's|\boutput_dir: .*|output_dir: '"$MODEL_PATH"'|' "$train_config"

    echo "[$(date)] Start training: Instruction tuning MODEL: $MODEL_NAME"
    $CONDA_PATH run -n ${ENV_NAME} --no-capture-output llamafactory-cli train ${train_config} 

    echo "Training has finished!"

    ##################################### SELECT_BEST_CHECKPOINT #####################################
    JSON_DATA_FILE="${MODEL_PATH}/trainer_log.jsonl"
    max_eval_accuracy="0.0"
    max_eval_accuracy_step=""

    while IFS= read -r line; do
    eval_accuracy=$(echo "$line" | jq -r '.eval_accuracy // empty')
    current_step=$(echo "$line" | jq -r '.current_steps // empty')

    if [ -z "$eval_accuracy" ]; then
        continue
    fi
    
    echo "eval_accuracy: ${eval_accuracy}, current_step: ${current_step}"
    
    if awk "BEGIN {exit !( $eval_accuracy > $max_eval_accuracy )}"; then
        max_eval_accuracy="$eval_accuracy"
        max_eval_accuracy_step="$current_step"
        echo "New maximum eval_accuracy: $max_eval_accuracy at step: $max_eval_accuracy_step"
    fi

    done < "$JSON_DATA_FILE"

    echo "Selected step with maximum accuracy: ${max_eval_accuracy_step}"

    ##################################### MERGE #####################################

    MERGE_LORA_PATH="$WORK_PATH/merge_lora"
    ADAPTER_NAME="$MODEL_PATH/checkpoint-${max_eval_accuracy_step}"

    OUTPUT_MERGE_FILE="${MERGE_LORA_PATH}/Instruct_${MODEL_NAME}.yaml"

    cat <<EOF > "$OUTPUT_MERGE_FILE"

    model_name_or_path: ${BASE_MODEL_PATH}
    adapter_name_or_path: ${ADAPTER_NAME}
    template: ${TEMPLATE}
    finetuning_type: lora

    export_dir: ${MERGE_MODEL_PATH}
    export_size: 2
    export_device: cpu
    export_legacy_format: false
EOF

    echo "The config file has been generated: $OUTPUT_MERGE_FILE"

    ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output llamafactory-cli export ${OUTPUT_MERGE_FILE}
    echo "Merge has finished!"
done

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

    ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output python ${INFERENCE_PATH}/Qwen2-VL/infer.py \
        --base_model_path $BASE_MODEL_PATH \
        --merge_model_path $MERGE_MODEL_PATH \
        --base_data_path $BASE_DATA_PATH \
        --dataset $DATASET_NAME \
        --results_path $RESULTS_PATH/$MODEL_NAME \
        --task $TASK_NAME \
        --device_ids $DEVICE_IDS

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

