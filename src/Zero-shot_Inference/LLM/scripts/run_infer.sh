#!/bin/bash

BASE_PATH="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/zhanghanlei"
BASE_DATA_PATH="$BASE_PATH/Datasets"
CONDA_PATH="$BASE_PATH/anaconda3/bin/conda"
INFERENCE_PATH="$BASE_PATH/MMLA_Code/Zero-shot_Inference/LLM"
RESULTS_PATH="$INFERENCE_PATH/results"
EVALUATION_PATH="$BASE_PATH/MMLA_Code/Zero-shot_Inference"

############ You can select one model for inference below. ############

# MODEL_NAMES=("Llama_3_1_8B_Instruct") 
# MODEL_NAMES=("Llama_3_2_1B_Instruct") 
# MODEL_NAMES=("Llama_3_2_3B_Instruct") 
# MODEL_NAMES=("Llama_3_8B_Instruct") 
# MODEL_NAMES=("Qwen2_0_5B_Instruct")
# MODEL_NAMES=("Qwen2_1_5B_Instruct")
# MODEL_NAMES=("Qwen2_7B_Instruct")
MODEL_NAMES=("Internlm2_5")

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
declare -A python_scripts
python_scripts['Llama_3_1_8B_Instruct']='infer.py'
python_scripts['Llama_3_2_1B_Instruct']='infer.py'
python_scripts['Llama_3_2_3B_Instruct']='infer.py'
python_scripts['Llama_3_8B_Instruct']='infer.py'
python_scripts['Qwen2_0_5B_Instruct']='infer.py'
python_scripts['Qwen2_1_5B_Instruct']='infer.py'
python_scripts['Qwen2_7B_Instruct']='infer.py'
python_scripts['Internlm2_5']='infer.py'

########################################

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    BASE_MODEL_PATH="$BASE_PATH/llm_checkpoints/$MODEL_NAME"
    DEVICE_IDS="${MODEL_DEVICES[$MODEL_NAME]}"

    for dataset in "${!task_map[@]}"; do
        DATASET_NAME=$dataset
        TASK_NAME="${task_map[$dataset]}"
        ENV_NAME="${ENV_NAMES[$MODEL_NAME]}"

        timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

        echo "[$(date)] Start zero-shot inference: $TASK_NAME - $DATASET_NAME MODEL: $MODEL_NAME"
        
        python_script="${python_scripts[$MODEL_NAME]}"
        
        ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output python ${INFERENCE_PATH}/${python_script} \
            --base_model_path $BASE_MODEL_PATH \
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
