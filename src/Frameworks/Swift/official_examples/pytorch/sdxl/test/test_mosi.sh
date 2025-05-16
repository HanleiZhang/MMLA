ckpt_dirs=(
  "/root/autodl-tmp/wpw/swift/output/MOSI/minicpm-v-v2_6-chat/v1-20241128-200430/checkpoint-380"
)

val_datasets=(
  "/root/autodl-tmp/wpw/swift/test/MOSI/test.jsonl"
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
