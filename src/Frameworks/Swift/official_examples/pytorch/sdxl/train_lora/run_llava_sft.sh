
train_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MELD/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/meld-da/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMPCAP-DA/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/Ch-sims/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-client/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-therapist/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MOSI/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MUSTARD/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/UR-FUNNY/train.jsonl" ""/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMOCAP/train.jsonl"")
eval_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MELD/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/meld-da/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMPCAP-DA/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/Ch-sims/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-client/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-therapist/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MOSI/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MUSTARD/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/UR-FUNNY/dev.jsonl"  "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMOCAP/dev.jsonl")

# train_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MELD/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/meld-da/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMPCAP-DA/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/Ch-sims/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-client/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-therapist/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MOSI/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MUSTARD/train.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/UR-FUNNY/train.jsonl" ""/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMOCAP/train.jsonl"")
# eval_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MIntRec2.0/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MELD/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/meld-da/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMPCAP-DA/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/Ch-sims/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-client/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-therapist/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MOSI/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MUSTARD/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/UR-FUNNY/dev.jsonl"  "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMOCAP/dev.jsonl")


# lora_target_module='^(llm|resampler)(?!.*(lm_head|output|emb|wte|shared)).*'
# 遍历所有的数据集
for index in ${!train_datasets[@]}; do
    train_dataset=${train_datasets[$index]}
    eval_dataset=${eval_datasets[$index]}
    dataset_name=$(basename $(dirname ${train_dataset}))
    # eval_name=$(basename $(dirname ${eval_dataset}))
    
    # 运行训练
    CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
      --model_type llava-next-video-7b-instruct \
      --model_id_or_path /root/autodl-tmp/model/LLaVA-Video-7B-Qwen2 \
      --template_type llava-next-video \
      --sft_type lora \
      --dataset ${train_dataset} \
      --val_dataset ${eval_dataset} \
      --max_length 1024 \
      --lora_rank 16 \
      --init_lora_weights True \
      --learning_rate 1e-4 \
      --num_train_epochs 6 \
      --gradient_accumulation_steps 16 \
      --eval_steps 50 \
      --save_steps 50 \
      --neftune_noise_alpha 5 \
      --eval_batch_size 1 \
      --deepspeed default-zero2 \
      --output_dir /root/autodl-tmp/model/wpw/output/LLaVA-Video-7B-Qwen2/${dataset_name} \

done
