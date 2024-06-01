CUDA_VISIBLE_DEVICES=5 python model_full_transformation.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/gpt-neo-125m-lora/checkpoint-80' \
    --output_dir './trained_models/full_model'
