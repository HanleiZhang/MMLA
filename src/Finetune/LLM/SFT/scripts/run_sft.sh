#!/bin/bash

MASTER_PORT=23457 

BASE_PATH="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/zhanghanlei"
BASE_DATA_PATH="$BASE_PATH/Datasets"
CONDA_PATH="$BASE_PATH/anaconda3/bin/conda"
WORK_PATH="$BASE_PATH/MMLA/src/Finetune/LLM/SFT"
INFERENCE_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference/LLM"
FRAMEWORK_PATH="$BASE_PATH/MMLA/src/Frameworks/LLaMA-Factory-main"
EVALUATION_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference"
RESULTS_PATH="$WORK_PATH/results"

MODEL_NAMES=("Internlm2_5" "Llama_3_1_8B_Instruct" "Llama_3_2_1B_Instruct" "Llama_3_2_3B_Instruct" "Llama_3_8B_Instruct" "Qwen2_0_5B_Instruct" "Qwen2_1_5B_Instruct" "Qwen2_7B_Instruct")
###############################
declare -A TEMPLATES

TEMPLATES["Llama_3_1_8B_Instruct"]="llama3"
TEMPLATES["Llama_3_2_1B_Instruct"]="llama3"
TEMPLATES["Llama_3_2_3B_Instruct"]="llama3"
TEMPLATES["Llama_3_8B_Instruct"]="llama3"
TEMPLATES["Qwen2_0_5B_Instruct"]="qwen"
TEMPLATES["Qwen2_1_5B_Instruct"]="qwen"
TEMPLATES["Qwen2_7B_Instruct"]="qwen"
TEMPLATES["Internlm2_5"]="intern2"

###############################
declare -A ENV_NAMES

ENV_NAMES["Llama_3_1_8B_Instruct"]="LLaMA_Factory"
ENV_NAMES["Llama_3_2_1B_Instruct"]="LLaMA_Factory"
ENV_NAMES["Llama_3_2_3B_Instruct"]="LLaMA_Factory"
ENV_NAMES["Llama_3_8B_Instruct"]="LLaMA_Factory"
ENV_NAMES["Qwen2_0_5B_Instruct"]="LLaMA_Factory"
ENV_NAMES["Qwen2_1_5B_Instruct"]="LLaMA_Factory"
ENV_NAMES["Qwen2_7B_Instruct"]="LLaMA_Factory"
ENV_NAMES["Internlm2_5"]="LLaMA_Factory"

###############################
declare -A MODEL_DEVICES

MODEL_DEVICES["Llama_3_1_8B_Instruct"]="0"
MODEL_DEVICES["Llama_3_2_1B_Instruct"]="0"
MODEL_DEVICES["Llama_3_2_3B_Instruct"]="0" 
MODEL_DEVICES["Llama_3_8B_Instruct"]="0" 
MODEL_DEVICES["Qwen2_0_5B_Instruct"]="0" 
MODEL_DEVICES["Qwen2_1_5B_Instruct"]="0"
MODEL_DEVICES["Qwen2_7B_Instruct"]="0"
MODEL_DEVICES["Internlm2_5"]="0"

#######################################
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

########################################

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    BASE_MODEL_PATH="$BASE_PATH/llm_checkpoints/$MODEL_NAME"
    DEVICE_IDS="${MODEL_DEVICES[$MODEL_NAME]}"

    for dataset in "${!task_map[@]}"; do
        DATASET_NAME=$dataset
        TASK_NAME="${task_map[$dataset]}"
        ENV_NAME="${ENV_NAMES[$MODEL_NAME]}"
        TEMPLATE="${TEMPLATES[$MODEL_NAME]}"

        train_data_json="${DATASET_NAME}_train_text_${TASK_NAME}_llamafactory"
        dev_data_json="${DATASET_NAME}_dev_text_${TASK_NAME}_llamafactory"

        MODEL_PATH="$BASE_PATH/models_sft/$DATASET_NAME/$TASK_NAME/${MODEL_NAME}_SFT"
        MERGE_MODEL_PATH="$BASE_PATH/merge_models_sft/$DATASET_NAME/$TASK_NAME/${MODEL_NAME}_SFT"

        timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

        echo "[$(date)] Start training: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"
        ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output llamafactory-cli train \
            --stage sft \
            --do_train True \
            --model_name_or_path $BASE_MODEL_PATH \
            --preprocessing_num_workers 16 \
            --finetuning_type lora \
            --template $TEMPLATE \
            --flash_attn auto \
            --dataset_dir $FRAMEWORK_PATH/data \
            --dataset $train_data_json \
            --per_device_train_batch_size 1 \
            --eval_dataset $dev_data_json \
            --per_device_eval_batch_size 1 \
            --eval_strategy steps \
            --eval_steps 20 \
            --output_dir $MODEL_PATH \
            --cutoff_len 1024 \
            --learning_rate 5e-5 \
            --num_train_epochs 8.0 \
            --max_samples 100000 \
            --gradient_accumulation_steps 4 \
            --lr_scheduler_type cosine \
            --max_grad_norm 1.0 \
            --logging_steps 1 \
            --save_steps 10 \
            --packing False \
            --report_to none \
            --bf16 True \
            --plot_loss True \
            --ddp_timeout 180000000 \
            --include_num_input_tokens_seen True \
            --lora_rank 16 \
            --lora_alpha 32 \
            --lora_target all \
            --overwrite_cache True \
            --overwrite_output_dir True \
            --compute_accuracy True
        
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

        OUTPUT_MERGE_FILE="${MERGE_LORA_PATH}/${DATASET_NAME}_${TASK_NAME}_${MODEL_NAME}.yaml"

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

        ##################################### INFERENCE #####################################

        echo "[$(date)] Start zero-shot inference: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"
        
        ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output python ${INFERENCE_PATH}/infer.py \
            --base_model_path $BASE_MODEL_PATH \
            --merge_model_path $MERGE_MODEL_PATH \
            --base_data_path $BASE_DATA_PATH \
            --dataset $DATASET_NAME \
            --results_path $RESULTS_PATH/$MODEL_NAME \
            --task $TASK_NAME \
            --device_ids $DEVICE_IDS  
        
        echo "Inference finished!"

        echo "[$(date)] Start evaluation: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"
        
        ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output python ${EVALUATION_PATH}/eval.py \
            --dataset $DATASET_NAME \
            --task $TASK_NAME \
            --model $MODEL_NAME \
            --results_path $RESULTS_PATH/$MODEL_NAME \
            --timestamp $timestamp \

        echo "Evaluation finished!"

    done
done