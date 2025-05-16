#!/bin/bash

# mgpu put_task --session_name yshzhu_nj_llm4_1x8 --cmd 'bash /mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/zhanghanlei/MMLA_Code/Zero-shot_Inference/MLLM/scripts/run_infer.sh'

BASE_PATH="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/zhanghanlei"
BASE_DATA_PATH="$BASE_PATH/Datasets"
CONDA_PATH="$BASE_PATH/anaconda3/bin/conda"
FRAMEWORK_PATH="$BASE_PATH/MMLA/src/Frameworks/Swift"
INFERENCE_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference/MLLM"
RESULTS_PATH="$INFERENCE_PATH/results"
EVALUATION_PATH="$BASE_PATH/MMLA/src/Zero-shot_Inference"

# MODEL_NAMES=("LLaVA-Video-72B-Qwen2") 
# MODEL_NAMES=("LLaVA-Video-7B-Qwen2") 
# MODEL_NAMES=("Qwen2-VL-7B-Instruct") 
# MODEL_NAMES=("Qwen2-VL-72B-Instruct") 
# MODEL_NAMES=("llava-onevision-qwen2-7b-ov-chat")
# MODEL_NAMES=("llava-onevision-qwen2-72b-ov-chat")
# MODEL_NAMES=("VideoLLaMA2-7B")
MODEL_NAMES=("MiniCPM-V-2_6")

###############################
declare -A ENV_NAMES

ENV_NAMES["Qwen2-VL-72B-Instruct"]="LLaMA_Factory"
ENV_NAMES["Qwen2-VL-7B-Instruct"]="LLaMA_Factory"
ENV_NAMES["LLaVA-Video-7B-Qwen2"]="LLaVA"
ENV_NAMES["LLaVA-Video-72B-Qwen2"]="LLaVA"
ENV_NAMES["llava-onevision-qwen2-7b-ov-chat"]="LLaVA"
ENV_NAMES["llava-onevision-qwen2-72b-ov-chat"]="LLaVA"
ENV_NAMES["VideoLLaMA2-7B"]="VideoLLaMA2"
ENV_NAMES["MiniCPM-V-2_6"]="MiniCPM"

###############################
declare -A MODEL_DEVICES

MODEL_DEVICES["Qwen2-VL-72B-Instruct"]="0,1,2,3,4,5,6,7"
MODEL_DEVICES["Qwen2-VL-7B-Instruct"]="0"
MODEL_DEVICES["LLaVA-Video-7B-Qwen2"]="0,1,2,3,4,5,6,7" 
MODEL_DEVICES["LLaVA-Video-72B-Qwen2"]="0,1,2,3,4,5,6,7" 
MODEL_DEVICES["llava-onevision-qwen2-7b-ov-chat"]="0,1,2,3,4,5,6,7" 
MODEL_DEVICES["llava-onevision-qwen2-72b-ov-chat"]="0,1,2,3,4,5,6,7"
MODEL_DEVICES["VideoLLaMA2-7B"]="0,1,2,3,4,5,6,7"
MODEL_DEVICES["MiniCPM-V-2_6"]="4"

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
python_scripts['LLaVA-Video-7B-Qwen2']='LLaVA-NeXT/infer.py'
python_scripts['LLaVA-Video-72B-Qwen2']='LLaVA-NeXT/infer.py'
python_scripts['llava-onevision-qwen2-7b-ov-chat']='LLaVA-NeXT/infer.py'
python_scripts['llava-onevision-qwen2-72b-ov-chat']='LLaVA-NeXT/infer.py'
python_scripts['Qwen2-VL-7B-Instruct']='Qwen2-VL/infer.py'
python_scripts['Qwen2-VL-72B-Instruct']='Qwen2-VL/infer.py'
python_scripts['VideoLLaMA2-7B']='VideoLLaMA2/infer.py'
python_scripts['MiniCPM-V-2_6']='MiniCPM-V-2_6/infer.py'

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
        
        if [ "$MODEL_NAME" == "MiniCPM-V-2_6" ]; then
            nproc_per_node=1
            CUDA_VISIBLE_DEVICES=${DEVICE_IDS} NPROC_PER_NODE=$nproc_per_node ${CONDA_PATH} run -n ${ENV_NAME} swift infer \
                --model $BASE_MODEL_PATH \
                --infer_backend pt \
                --val_dataset $FRAMEWORK_PATH/video_prompt_data/${DATASET_NAME}_test.jsonl \
                --result_path $RESULTS_PATH/$MODEL_NAME/${DATASET_NAME}_${TASK_NAME}_result.jsonl
        else
            ${CONDA_PATH} run -n ${ENV_NAME} --no-capture-output python ${INFERENCE_PATH}/${python_script} \
                --base_model_path $BASE_MODEL_PATH \
                --base_data_path $BASE_DATA_PATH \
                --dataset $DATASET_NAME \
                --results_path $RESULTS_PATH/$MODEL_NAME \
                --task $TASK_NAME \
                --device_ids $DEVICE_IDS  
        fi
        
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
