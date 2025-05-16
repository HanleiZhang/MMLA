# CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
#   --model_type minicpm-v-v2_6-chat \
#   --model_id_or_path /root/autodl-tmp/model/wpw/model/minicpm-v \
#   --sft_type lora \
#   --deepspeed default-zero2 \
#   --dataset /root/autodl-tmp/wpw/pre_dataset/swift/video_prompt_data/MIntRec/MIntRec_train_intent_llamafactory.json.jsonl \
#   --val_dataset /root/autodl-tmp/wpw/pre_dataset/swift/video_prompt_data/MIntRec/MIntRec_dev_intent_llamafactory.json.jsonl \
#   --output_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/ \



# """
# CUDA_VISIBLE_DEVICES=0,1,2,3  NPROC_PER_NODE=4 swift sft --model_id_or_path 'OpenBMB/MiniCPM-V-2_6' --template_type 'minicpm-v-v2_6' --system 'You are a helpful assistant.' --dataset 'zhihu-kol-filtered' --lora_target_modules '^(llm|resampler)(?!.*(lm_head|output|emb|wte|shared)).*' --lora_rank '16' --init_lora_weights 'True' --learning_rate '1e-4' --gradient_accumulation_steps '16' --eval_steps '50' --save_steps '50' --eval_batch_size '1'  --deepspeed default-zero2  --add_output_dir_suffix False --output_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/minicpm-v-v2_6-chat/v3-20241111-164240 --logging_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/minicpm-v-v2_6-chat/v3-20241111-164240/runs --ignore_args_error True > /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/minicpm-v-v2_6-chat/v3-20241111-164240/runs/run.log 2>&1 &

# """
# # lora_target_module='\^(llm|resampler)(?!.*(lm_head|output|emb|wte|shared)).*'

# CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 nohup swift sft \
# 	--model_id_or_path /root/autodl-tmp/model/wpw/model/minicpm-v \ 
# 	--template_type minicpm-v-v2_6 \ 
# 	--sft_type lora \ 
# 	--system You are a helpful assistant. \ 
# 	--dataset zhihu-kol-filtered \ 
# 	--val_dataset /root/autodl-tmp/wpw/pre_dataset/swift/video_prompt_data/MIntRec/MIntRec_dev_intent_llamafactory.json.jsonl \
# 	--max_length 1024 \ 
# 	--lora_target_modules '\^(llm|resampler)(?!.*(lm_head|output|emb|wte|shared)).*' 
# 	--lora_rank 16 \
# 	--init_lora_weights True \ 
# 	--learning_rate 1e-4 \ 
# 	--num_train_epochs 6.0 \ 
# 	--use_flash_attn True \
# 	--gradient_accumulation_steps 16 \ 
# 	--eval_steps 100 \ 
# 	--save_steps 100 \ 
# 	--neftune_noise_alpha 5 \ 
# 	--eval_batch_size 1 \  
# 	--deepspeed default-zero2 \ 
# 	--add_output_dir_suffix False 
# 	--output_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/minicpm-v-v2_6-chat/v4-20241111-170408 \
# 	--logging_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/minicpm-v-v2_6-chat/v4-20241111-170408/runs \ 
# 	--ignore_args_error True > /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/minicpm-v-v2_6-chat/v4-20241111-170408/runs/run.log 2>&1 &
# train_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MELD/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/meld-da/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMOCAP/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMPCAP-DA/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/Ch-sims/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-client/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-therapist/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MOSI/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MUSTARD/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/UR-FUNNY/train.jsonl")
# eval_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MELD/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/meld-da/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMOCAP/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMPCAP-DA/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/Ch-sims/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-client/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-therapist/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MOSI/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MUSTARD/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/UR-FUNNY/dev.jsonl")

# lora_target_module='^(llm|resampler)(?!.*(lm_head|output|emb|wte|shared)).*'
# # 遍历所有的数据集
# for index in ${!train_datasets[@]}; do
#     train_dataset=${train_datasets[$index]}
#     eval_dataset=${eval_datasets[$index]}
    
#     # 运行训练
#     CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
#       --model_type minicpm-v-v2_6-chat \
#       --model_id_or_path /root/autodl-tmp/model/wpw/model/minicpm-v \
#       --sft_type lora \
#       --system "You are a helpful assistant." \
#       --dataset ${train_dataset} \
#       --val_dataset ${eval_dataset} \
#       --max_length 1024 \
#       --lora_target_modules ${lora_target_module} \
#       --lora_rank 16 \
#       --init_lora_weights True \
#       --learning_rate 1e-4 \
#       --num_train_epochs 6 \
#       --use_flash_attn True \
#       --gradient_accumulation_steps 16 \
#       --eval_steps 100 \
#       --save_steps 100 \
#       --neftune_noise_alpha 5 \
#       --eval_batch_size 1 \
#       --deepspeed default-zero2 \
#       --add_output_dir_suffix False \
#       --output_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/$(basename ${train_dataset}) \
#       --logging_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/$(basename ${train_dataset}) \   
# done

# CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
#   --model_type minicpm-v-v2_6-chat \
#   --model_id_or_path /root/autodl-tmp/model/wpw/model/minicpm-v \
#   --sft_type lora \
#   --deepspeed default-zero2 \
#   --dataset /root/autodl-tmp/wpw/pre_dataset/swift/video_prompt_data/MIntRec/MIntRec_train_intent_llamafactory.json.jsonl \
#   --val_dataset /root/autodl-tmp/wpw/pre_dataset/swift/video_prompt_data/MIntRec/MIntRec_dev_intent_llamafactory.json.jsonl \
#   --output_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/ \
# train_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MELD/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/meld-da/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMPCAP-DA/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/Ch-sims/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-client/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-therapist/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MOSI/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MUSTARD/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/UR-FUNNY/train.jsonl" ""/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMOCAP/train.jsonl"")
# eval_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MELD/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/meld-da/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMPCAP-DA/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/Ch-sims/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-client/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-therapist/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MOSI/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MUSTARD/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/UR-FUNNY/dev.jsonl"  "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMOCAP/dev.jsonl")

train_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/train.jsonl")
eval_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/dev.jsonl")


# lora_target_module='^(llm|resampler)(?!.*(lm_head|output|emb|wte|shared)).*'
# 遍历所有的数据集
for index in ${!train_datasets[@]}; do
    train_dataset=${train_datasets[$index]}
    eval_dataset=${eval_datasets[$index]}
    dataset_name=$(basename $(dirname ${train_dataset}))
    # eval_name=$(basename $(dirname ${eval_dataset}))
    
    # 运行训练
    CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
      --model_type minicpm-v-v2_6-chat \
      --model_id_or_path /root/autodl-tmp/model/wpw/model/minicpm-v \
      --sft_type lora \
      --dataset ${train_dataset} \
      --val_dataset ${eval_dataset} \
      --max_length 1024 \
      --lora_rank 16 \
      --init_lora_weights True \
      --learning_rate 1e-4 \
      --num_train_epochs 4 \
      --use_flash_attn false \
      --gradient_accumulation_steps 8 \
      --eval_steps 100 \
      --save_steps 100 \
      --neftune_noise_alpha 4 \
      --eval_batch_size 1 \
      --deepspeed default-zero2 \
      --output_dir /root/autodl-tmp/model/wpw/output/minicpm_MIntRec2 \
      
done
