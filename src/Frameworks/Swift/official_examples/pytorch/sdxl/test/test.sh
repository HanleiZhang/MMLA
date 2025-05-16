# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --ckpt_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/minicpm-v-v2_6-chat/v2-20241111-151810/checkpoint-83 \
#     
#     --
#   --deepspeed default-zero2 \
# ckpt_dirs=("/root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/AnnoMi-therapist/minicpm-v-v2_6-chat/v0-20241113-022951/checkpoint-200" "/root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/IEMPCAP-DA/minicpm-v-v2_6-chat/v0-20241112-213237/checkpoint-500" "/root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/MOSI/minicpm-v-v2_6-chat/v0-20241113-042802/checkpoint-100" "/root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/MUSTARD/minicpm-v-v2_6-chat/v0-20241113-045303/checkpoint-65" "/root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/UR-FUNNY/minicpm-v-v2_6-chat/v0-20241113-050242/checkpoint-200")
# val_datasets=("/root/autodl-tmp/wpw/new_ft_cpm-v/test/AnnoMi-therapist/test_communication_behavior.json.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/test/IEMOCAP-DA/test_dialogue_act.json.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/test/MOSI/test_sentiment.json.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/test/MUSTARD/test_speaking_style.json.jsonl" "/root/autodl-tmp/wpw/new_ft_cpm-v/test/UR-FUNNY/test_speaking_style.json.jsonl")


# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --ckpt_dir /root/autodl-tmp/wpw/new_ft_cpm-v/swift/output/tested/MIntRec/checkpoint-100 \
#     --load_dataset_config true --merge_lora true \
#     --val_dataset /root/autodl-tmp/wpw/new_ft_cpm-v/test/MIntRec/test_intent.jsonl 
# CUDA_VISIBLE_DEVICES=0 swift infer \
#   --model_type minicpm-v-v2_6-chat \
#   --model_id_or_path /root/autodl-tmp/model/wpw/model/minicpm-v \
#   --load_dataset_config true --merge_lora true \
#   --ckpt_dir /root/autodl-tmp/wpw/swift/output/AnnoMi-therapist/minicpm-v-v2_6-chat/v0-20241113-022951/checkpoint-200 \
#   --sft_type lora \
#   --val_dataset /root/autodl-tmp/wpw/swift/test/AnnoMi-therapist/test_communication_behavior.jsonl \


ckpt_dirs=(
  "/root/autodl-tmp/wpw/swift/output/Ch-sims/minicpm-v-v2_6-chat/v45-20241115-031533/checkpoint-700"
  "/root/autodl-tmp/wpw/swift/output/IEMPCAP-DA/minicpm-v-v2_6-chat/v0-20241112-213237/checkpoint-500"
  "/root/autodl-tmp/wpw/swift/output/MOSI/minicpm-v-v2_6-chat/v0-20241113-042802/checkpoint-100"
  "/root/autodl-tmp/wpw/swift/output/MUSTARD/minicpm-v-v2_6-chat/v0-20241113-045303/checkpoint-65"
  "/root/autodl-tmp/wpw/swift/output/UR-FUNNY/minicpm-v-v2_6-chat/v0-20241113-050242/checkpoint-200"
  "/root/autodl-tmp/wpw/swift/output/IEMOCAP/minicpm-v-v2_6-chat/v2-20241113-081144/checkpoint-300"
)

val_datasets=(
  "/root/autodl-tmp/wpw/swift/test/Ch-sims/test_sentiment.json.jsonl"
  "/root/autodl-tmp/wpw/swift/test/IEMOCAP/IEMOCAP_test_emotion.json.jsonl"
  "/root/autodl-tmp/wpw/swift/test/IEMOCAP-DA/test_dialogue_act.json.jsonl"
  "/root/autodl-tmp/wpw/swift/test/MOSI/test_sentiment.json.jsonl"
  "/root/autodl-tmp/wpw/swift/test/MUSTARD/test_speaking_style.json.jsonl"
  "/root/autodl-tmp/wpw/swift/test/UR-FUNNY/test_speaking_style.json.jsonl"
  "/root/autodl-tmp/wpw/swift/test/IEMOCAP/IEMOCAP_test_emotion.json.jsonl"
)

# 遍历ckpt_dirs和val_datasets
for i in ${!ckpt_dirs[@]}; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 swift infer \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path /root/autodl-tmp/model/wpw/model/minicpm-v \
    --load_dataset_config true --merge_lora true \
    --ckpt_dir ${ckpt_dirs[$i]} \
    --sft_type lora \
    --val_dataset ${val_datasets[$i]}
done
