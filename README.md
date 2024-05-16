<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> 
<img src="picture.jpg" width="73" height="114">
<p> LLM-LeakageAuditor</p>
</h1>

<h3 align="center">
    <p>LLM-LeakageAuditor: A Privacy Assessment on a Family of Parameter-Efficient Fine-Tuning for Large Language Models </p>
</h3>
LLM-LeakageAuditor is an easy-to-use framework that integrates various adapters into LLMs and can execute adapter-based PEFT methods of LLMs for different tasks and asssess the privacy leakage risk.

The framework includes state-of-the-art open-access LLMs: LLaMa, OPT, BLOOM, and GPT-J, as well as widely used adapters such as Bottleneck adapters, Parallel adapters, and LoRA.

Supported Adapters:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. AdapterH: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)
3. AdapterP: [GMAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://arxiv.org/pdf/2005.00052.pdf)
4. Parallel: [TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING](https://arxiv.org/pdf/2110.04366.pdf)
5. Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
6. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
7. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 

## Latest News 🔥🔥

## Special Announcement
The `math_10k.json` data is collected with the training sets of GSM8K, MAWPS, and AQuA(1000 examples). However, MAWPS consists of AddSub, MultiArith, SingleOp, SingleEq, SimulEq-S, SimulEq-L. Thus, we can't utilize MultiArith, AddSub, and SingleEq as evaluation benchmarks with models trained with `math_10k.json`. We evaluate the PEFT methods on the MAWPS test set instead, and the result table has been updated (The findings in the paper are consistent). Furthermore, two variations of `math_10k.json` have been uploaded, `math_7K.json` where the MAWPS samples have been deleted, and `math_14k.json` where the MAWPS samples have been deleted as well and we combine ChatGPT and GPT-4 rationales. Sincerely apologize for any inconvenience!

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Set environment variables, or modify the files referencing `BASE_MODEL`:

```bash
# Files referencing `BASE_MODEL`
# export_hf_checkpoint.py
# export_state_dict_checkpoint.py

export BASE_MODEL=yahma/llama-7b-hf
```

Both `finetune.py` and `generate.py` use `--base_model` flag as shown further below.

3. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

## Training(finetune.py)

This file contains some code related to prompt construction and tokenization.In this file, specify different adapters and different sets of data, so that different models can be trained. 

Example usage for multiple GPUs:

```bash
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'math_10k.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

The `math_10k.json` data is collected with the training sets of GSM8K, MAWPS, and AQuA(1000 examples). `yahma/llama-7b-hf` is a base model, LLaMa-7B. Add `lora` adapter to this model.

Example usage for Single GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'math_10k.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

Moreover, you can use `--use_gradient_checkpointing` to save more GPU memory, but it will increase the training time.

To use the AdapterH, just add the following arguments:

```bash
--adapter_name bottleneck # use the bottleneck adapter, refers to AdapterH in the result table
```

To use the AdapterP, just add the following arguments:

```bash
--adapter_name bottleneck 
--use_adapterp  # use the AdapterP, refers to AdapterP in the result table
```

To use parallel adapter, just add the following arguments:

```bash
--adapter_name bottleneck
--use_parallel_adapter
```

Note that, In order to facilitate INT8 training of large models with parallel adapters, we have adopted a technique whereby the parallel adapter layers are incorporated into multi-head attention layers and MLP layers, in parallel with Linear layers. It is different from [Hu et al. (2021)](https://arxiv.org/pdf/2106.09685.pdf). 

## Inference (generate.py)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `'./trained_models/llama-lora'` , and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.
Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun generate.py \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
```

## Evaluation (evaluate.py)

To evaluate the performance of the finetuned model on the Arithmetic Reasoning tasks, you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py 
    --model LLaMA-7B \ #specify the base model
    --adapter LoRA \   #specify the adapter name ["LoRA", "AdapterH", "AdapterP", "Parallel"， "Scaled_Parallel""]
    --dataset SVAMP \  #specify the test dataset
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
```

<!-- ## Resource Consumption

There is a table of resouce needed for different adapters, which contains Trainable Parameters, GPU RAM Usage, and Fine-tuning Time on the Arithmetic Reasoning dataset `math_10k.json`

Hyper-parameter setting: num_epochs=3, lora_r=8, lora_alpha=16, bottleneck_size=256

Models: LLaMA-13B, LLaMA-7B, BLOOM-6.7B, GPT-j-6B
Dataset: 3.2K math word problems

Hardware: 2*3090 GPUs

| Model                 | Trainable Parameters | GPU RAM Usage | Fine-tuning Time |
|-----------------------|----------------------|---------------|------------------|
| LLaMA-7B-LoRA         | 4.2M                 | 18GB          |     4h           | 
| LLaMA-7B-AdapterH     | 200M                 | 22GB          |     4h           | 
| LLaMA-7B-AdapterP     | 200M                 | 22GB          |     4h           | 
| LLaMA-7B-Parallel     | 200M                 | 22GB          |     4h           |  -->


### PrivacyAuditor support matrix
This metrix shows whether different models can use LoRA,AdapterH,AdapterP,Parallel and Scaled Parallel adapters.

| Adapter      | LoRA | AdapterH | AdapterP | Parallel| Prefix Tuning	|P-Tuning|Prompt Tuning|
|--------------|-------|-------|----------|-------|-------|-------|-------|
| LLaMA        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| BLOOM        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     | 
| GPT-J        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| OPT          | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| GPT-2        | ✅     | 🔧Developing | 🔧Developing|🔧Developing | ✅     | ✅     | ✅     | 
| GPT-Neo      | ✅     | ✅     | ✅        | ✅    | ✅     | ✅     | ✅     | 
| GPT-NeoX-20B | ✅     | 🔧Developing | 🔧Developing|🔧Developing | ✅     | ✅     | ✅     |
| ChatGLM      | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     | 


### TODO List
- [x] Add Fine-tuning Approach: AdapterH
- [x] Add Fine-tuning Approach: AdapterP
- [x] Add Fine-tuning Approach: Parallel Adapter
- [ ] Support MIAs 


## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [Adapter-Transformer](https://github.com/adapter-hub/adapter-transformers), [Alpaca-lora](https://github.com/tloen/alpaca-lora), [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters). Thanks for their wonderful works. Additionally, we thank DONG Shan and [dream.ai](https://dream.ai/create) for the exceptional logo design, which has added immense value to our project. 
