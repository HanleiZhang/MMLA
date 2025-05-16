# datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/Ch-sims/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-client/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/AnnoMi-therapist/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MOSI/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/MUSTARD/dev.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/UR-FUNNY/dev.jsonl"  "/root/autodl-tmp/wpw/new_ft_cpm-v/video_prompt_data/IEMOCAP/dev.jsonl")

train_datasets=("/root/autodl-tmp/wpw/swift/video_prompt_data/MOSI/train.jsonl")
eval_datasets=("/root/autodl-tmp/wpw/swift/video_prompt_data/MOSI/dev.jsonl")
#       --deepspeed default-zero2 \
#    CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=1 swift sft \
#       --neftune_noise_alpha 5 \
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
      --num_train_epochs 6 \
      --use_flash_attn false \
      --gradient_accumulation_steps 1 \
      --eval_steps 20 \
      --save_steps 20 \
      --eval_batch_size 1 \
      --output_dir /root/autodl-tmp/wpw/swift/output/${dataset_name} \

done
