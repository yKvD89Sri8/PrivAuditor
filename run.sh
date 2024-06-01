CUDA_VISIBLE_DEVICES=5 python finetune.py \
  --base_model 'EleutherAI/gpt-neo-125m' \
  --data_path 'ft-training_set/math_10k.json' \
  --output_dir './trained_models/gpt-neo-125m-lora' \
  --batch_size 16 \
  --micro_batch_size 2 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --eval_step 16 \
  --save_step 16 \
  --use_gradient_checkpointing \
  --adapter_name lora
