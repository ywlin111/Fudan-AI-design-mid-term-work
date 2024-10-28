# Fudan-AI-design-mid-term-work
#1、环境安装
使用py310环境，Torch2.4.0，单块3090显卡。主要依赖包：
```python
pip install deepspeed -U
pip install transformers -U
pip install accelerate==0.34.2
```
安装llamafactory-cli命令
```python
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
#2、数据
直接使用adgen开源数据集
在LLaMA-Factory的Dataset.info中定义如下：
```python
"adgen_train": {
    "hf_hub_url": "HasturOfficial/adgen",
    "ms_hub_url": "AI-ModelScope/adgen",
    "split": "train",
    "columns": {
      "prompt": "content",
      "response": "summary"
    }
  }
```
数据集的大致内容截图：

![Image text](https://github.com/ywlin111/Fudan-AI-design-mid-term-work/dataset.png）
当输入为特定格式的精简描述时，预期对应的输出应该是基于描述扩写的流畅句子，内容基本覆盖原有信息。
#3、训练配置
基础模型使用Qwen2.5-0.5B-Instruct，进行LoRA微调，使用deepspeed/ds_z0_config.json作为加速配置，训练3个Epoch。
```python
### model
model_name_or_path: /opt/xxx/models/Qwen2.5-0.5B-Instruct/


### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
deepspeed: examples/deepspeed/ds_z0_config.json

### dataset
dataset: adgen_train
template: qwen
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/Qwen2.5-0.5B-Instruct/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```
#4、训练和测试脚本
##4.1 训练
训练脚本：
```python
#!/bin/bash


export FORCE_TORCHRUN=1
export CUDA_VISIBLE_DEVICES="4"

llamafactory-cli train examples/train_lora/qwen_lora_sft_ds0.yaml
```
训练过程：
```python
(py310) [root@hpc-009 LLaMA-Factory]# bash ./run_sft_train.sh
[2024-10-22 20:04:40,073] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
WARNING 10-22 20:04:42 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
10/22/2024 20:04:45 - INFO - llamafactory.cli - Initializing distributed tasks at: 127.0.0.1:20878
[2024-10-22 20:04:52,861] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-10-22 20:04:55,672] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-22 20:04:55,672] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
10/22/2024 20:04:55 - WARNING - llamafactory.hparams.parser - `ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.
10/22/2024 20:04:55 - INFO - llamafactory.hparams.parser - Process rank: 0, device: cuda:0, n_gpu: 1, distributed training: True, compute dtype: torch.bfloat16
[INFO|configuration_utils.py:673] 2024-10-22 20:04:55,759 >> loading configuration file /opt/xxx/models/Qwen2.5-0.5B-Instruct/config.json
[INFO|configuration_utils.py:742] 2024-10-22 20:04:55,760 >> Model config Qwen2Config {
  "_name_or_path": "/opt/xxx/models/Qwen2.5-0.5B-Instruct/",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:55,762 >> loading file vocab.json
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:55,762 >> loading file merges.txt
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:55,762 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:55,762 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:55,762 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:55,762 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2470] 2024-10-22 20:04:56,146 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|configuration_utils.py:673] 2024-10-22 20:04:56,147 >> loading configuration file /opt/xxx/models/Qwen2.5-0.5B-Instruct/config.json
[INFO|configuration_utils.py:742] 2024-10-22 20:04:56,148 >> Model config Qwen2Config {
  "_name_or_path": "/opt/xxx/models/Qwen2.5-0.5B-Instruct/",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:56,149 >> loading file vocab.json
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:56,149 >> loading file merges.txt
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:56,149 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:56,149 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:56,149 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2204] 2024-10-22 20:04:56,149 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2470] 2024-10-22 20:04:56,709 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
10/22/2024 20:04:56 - WARNING - llamafactory.model.loader - Processor was not found: 'Qwen2Config' object has no attribute 'vision_config'.
10/22/2024 20:04:56 - INFO - llamafactory.data.template - Replace eos token: <|im_end|>
10/22/2024 20:04:56 - INFO - llamafactory.data.loader - Loading dataset HasturOfficial/adgen...
Converting format of dataset (num_proc=16): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 3951.58 examples/s]
Running tokenizer on dataset (num_proc=16): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:02<00:00, 334.45 examples/s]
training example:
input_ids:
[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 31905, 2, 102693, 9, 40301, 24300, 2, 109285, 9, 104040, 2, 110803, 9, 108108, 2, 108236, 9, 102693, 24300, 2, 100607, 100447, 102693, 151645, 198, 151644, 77091, 198, 109285, 9370, 100607, 100447, 102693, 115950, 100672, 99544, 99742, 100411, 3837, 102152, 104070, 111949, 101421, 64355, 99242, 1773, 104163, 52801, 99621, 104070, 3837, 100165, 104297, 99621, 20221, 100447, 45861, 17, 72261, 105005, 109285, 9370, 102693, 100447, 3837, 109656, 106515, 99894, 30709, 26232, 44934, 103924, 1773, 17447, 95256, 99411, 33071, 99795, 16530, 103089, 62963, 3837, 110240, 99396, 100061, 102476, 99934, 101099, 41362, 98650, 102321, 102321, 113572, 1773, 38176, 99278, 99659, 100649, 70500, 110973, 3837, 97706, 99258, 114658, 105171, 98650, 108858, 1773, 117656, 108236, 100155, 99707, 100155, 46451, 9370, 3837, 110803, 112320, 17340, 1773, 102284, 105353, 106665, 9370, 3837, 57218, 113233, 100775, 31838, 104401, 9370, 104040, 104037, 94443, 99572, 105297, 1773, 151645]
inputs:
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤<|im_end|>
<|im_start|>assistant
宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。毕竟好穿时尚，谁都能穿出腿长2米的效果宽松的裤腿，当然是遮肉小能手啊。上身随性自然不拘束，面料亲肤舒适贴身体验感棒棒哒。系带部分增加设计看点，还让单品的设计感更强。腿部线条若隐若现的，性感撩人。颜色敲温柔的，与裤子本身所呈现的风格有点反差萌。<|im_end|>
label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 109285, 9370, 100607, 100447, 102693, 115950, 100672, 99544, 99742, 100411, 3837, 102152, 104070, 111949, 101421, 64355, 99242, 1773, 104163, 52801, 99621, 104070, 3837, 100165, 104297, 99621, 20221, 100447, 45861, 17, 72261, 105005, 109285, 9370, 102693, 100447, 3837, 109656, 106515, 99894, 30709, 26232, 44934, 103924, 1773, 17447, 95256, 99411, 33071, 99795, 16530, 103089, 62963, 3837, 110240, 99396, 100061, 102476, 99934, 101099, 41362, 98650, 102321, 102321, 113572, 1773, 38176, 99278, 99659, 100649, 70500, 110973, 3837, 97706, 99258, 114658, 105171, 98650, 108858, 1773, 117656, 108236, 100155, 99707, 100155, 46451, 9370, 3837, 110803, 112320, 17340, 1773, 102284, 105353, 106665, 9370, 3837, 57218, 113233, 100775, 31838, 104401, 9370, 104040, 104037, 94443, 99572, 105297, 1773, 151645]
labels:
宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。毕竟好穿时尚，谁都能穿出腿长2米的效果宽松的裤腿，当然是遮肉小能手啊。上身随性自然不拘束，面料亲肤舒适贴身体验感棒棒哒。系带部分增加设计看点，还让单品的设计感更强。腿部线条若隐若现的，性感撩人。颜色敲温柔的，与裤子本身所呈现的风格有点反差萌。<|im_end|>
[INFO|configuration_utils.py:673] 2024-10-22 20:05:06,641 >> loading configuration file /opt/xxx/models/Qwen2.5-0.5B-Instruct/config.json
[INFO|configuration_utils.py:742] 2024-10-22 20:05:06,643 >> Model config Qwen2Config {
  "_name_or_path": "/opt/xxx/models/Qwen2.5-0.5B-Instruct/",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|modeling_utils.py:3729] 2024-10-22 20:05:06,670 >> loading weights file /opt/xxx/models/Qwen2.5-0.5B-Instruct/model.safetensors
[INFO|modeling_utils.py:1622] 2024-10-22 20:05:06,679 >> Instantiating Qwen2ForCausalLM model under default dtype torch.bfloat16.
[INFO|configuration_utils.py:1099] 2024-10-22 20:05:06,681 >> Generate config GenerationConfig {
  "bos_token_id": 151643,
  "eos_token_id": 151645
}

[INFO|modeling_utils.py:4574] 2024-10-22 20:05:07,746 >> All model checkpoint weights were used when initializing Qwen2ForCausalLM.

[INFO|modeling_utils.py:4582] 2024-10-22 20:05:07,746 >> All the weights of Qwen2ForCausalLM were initialized from the model checkpoint at /opt/xxx/models/Qwen2.5-0.5B-Instruct/.
If your task is similar to the task the model of the checkpoint was trained on, you can already use Qwen2ForCausalLM for predictions without further training.
[INFO|configuration_utils.py:1052] 2024-10-22 20:05:07,748 >> loading configuration file /opt/xxx/models/Qwen2.5-0.5B-Instruct/generation_config.json
[INFO|configuration_utils.py:1099] 2024-10-22 20:05:07,749 >> Generate config GenerationConfig {
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "pad_token_id": 151643,
  "repetition_penalty": 1.1,
  "temperature": 0.7,
  "top_k": 20,
  "top_p": 0.8
}

10/22/2024 20:05:07 - INFO - llamafactory.model.model_utils.checkpointing - Gradient checkpointing enabled.
10/22/2024 20:05:07 - INFO - llamafactory.model.model_utils.attention - Using torch SDPA for faster training and inference.
10/22/2024 20:05:07 - INFO - llamafactory.model.adapter - Upcasting trainable params to float32.
10/22/2024 20:05:07 - INFO - llamafactory.model.adapter - Fine-tuning method: LoRA
10/22/2024 20:05:07 - INFO - llamafactory.model.model_utils.misc - Found linear modules: o_proj,v_proj,down_proj,k_proj,q_proj,gate_proj,up_proj
10/22/2024 20:05:08 - INFO - llamafactory.model.loader - trainable params: 4,399,104 || all params: 498,431,872 || trainable%: 0.8826
Detected kernel version 3.10.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
[INFO|trainer.py:667] 2024-10-22 20:05:08,062 >> Using auto half precision backend
[2024-10-22 20:05:08,481] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed info: version=0.15.2, git-hash=unknown, git-branch=unknown
[2024-10-22 20:05:08,481] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 1
[2024-10-22 20:05:08,933] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2024-10-22 20:05:08,938] [INFO] [logging.py:96:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2024-10-22 20:05:08,938] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
[2024-10-22 20:05:08,984] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = AdamW
[2024-10-22 20:05:08,984] [INFO] [logging.py:96:log_dist] [Rank 0] Creating BF16 optimizer
[2024-10-22 20:05:09,211] [INFO] [utils.py:781:see_memory_usage] begin bf16_optimizer
[2024-10-22 20:05:09,212] [INFO] [utils.py:782:see_memory_usage] MA 0.94 GB         Max_MA 0.94 GB         CA 0.98 GB         Max_CA 1 GB
[2024-10-22 20:05:09,212] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 41.11 GB, percent = 16.4%
[2024-10-22 20:05:09,442] [INFO] [utils.py:781:see_memory_usage] before initializing group 0
[2024-10-22 20:05:09,443] [INFO] [utils.py:782:see_memory_usage] MA 0.94 GB         Max_MA 0.94 GB         CA 0.98 GB         Max_CA 1 GB
[2024-10-22 20:05:09,443] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 41.11 GB, percent = 16.4%
[2024-10-22 20:05:09,677] [INFO] [utils.py:781:see_memory_usage] after initializing group 0
[2024-10-22 20:05:09,677] [INFO] [utils.py:782:see_memory_usage] MA 0.97 GB         Max_MA 0.97 GB         CA 1.04 GB         Max_CA 1 GB
[2024-10-22 20:05:09,678] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 41.12 GB, percent = 16.4%
[2024-10-22 20:05:09,898] [INFO] [utils.py:781:see_memory_usage] end bf16_ optimizer
[2024-10-22 20:05:09,899] [INFO] [utils.py:782:see_memory_usage] MA 0.97 GB         Max_MA 0.97 GB         CA 1.04 GB         Max_CA 1 GB
[2024-10-22 20:05:09,899] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 41.12 GB, percent = 16.4%
[2024-10-22 20:05:09,899] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = BF16_Optimizer
[2024-10-22 20:05:09,899] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using configured LR scheduler = None
[2024-10-22 20:05:09,899] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = None
[2024-10-22 20:05:09,899] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.0], mom=[(0.9, 0.999)]
[2024-10-22 20:05:09,904] [INFO] [config.py:999:print] DeepSpeedEngine configuration:
[2024-10-22 20:05:09,905] [INFO] [config.py:1003:print]   activation_checkpointing_config  {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
[2024-10-22 20:05:09,905] [INFO] [config.py:1003:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True, 'use_gds': False}
[2024-10-22 20:05:09,905] [INFO] [config.py:1003:print]   amp_enabled .................. False
[2024-10-22 20:05:09,905] [INFO] [config.py:1003:print]   amp_params ................... False
[2024-10-22 20:05:09,905] [INFO] [config.py:1003:print]   autotuning_config ............ {
    "enabled": false,
    "start_step": null,
    "end_step": null,
    "metric_path": null,
    "arg_mappings": null,
    "metric": "throughput",
    "model_info": null,
    "results_dir": "autotuning_results",
    "exps_dir": "autotuning_exps",
    "overwrite": true,
    "fast": true,
    "start_profile_step": 3,
    "end_profile_step": 5,
    "tuner_type": "gridsearch",
    "tuner_early_stopping": 5,
    "tuner_num_trials": 50,
    "model_info_path": null,
    "mp_size": 1,
    "max_train_batch_size": null,
    "min_train_batch_size": 1,
    "max_train_micro_batch_size_per_gpu": 1.024000e+03,
    "min_train_micro_batch_size_per_gpu": 1,
    "num_tuning_micro_batch_sizes": 3
}
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   bfloat16_enabled ............. True
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   bfloat16_immediate_grad_update  False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   checkpoint_parallel_write_pipeline  False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   checkpoint_tag_validation_enabled  True
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   checkpoint_tag_validation_fail  False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x7f7ca4380be0>
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   communication_data_type ...... None
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   curriculum_enabled_legacy .... False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   curriculum_params_legacy ..... False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   data_efficiency_enabled ...... False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   dataloader_drop_last ......... False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   disable_allgather ............ False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   dump_state ................... False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   dynamic_loss_scale_args ...... None
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   eigenvalue_enabled ........... False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   eigenvalue_gas_boundary_resolution  1
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   eigenvalue_layer_num ......... 0
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   eigenvalue_max_iter .......... 100
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   eigenvalue_stability ......... 1e-06
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   eigenvalue_tol ............... 0.01
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   eigenvalue_verbose ........... False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   elasticity_enabled ........... False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   flops_profiler_config ........ {
    "enabled": false,
    "recompute_fwd_factor": 0.0,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
}
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   fp16_auto_cast ............... None
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   fp16_enabled ................. False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   fp16_master_weights_and_gradients  False
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   global_rank .................. 0
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   grad_accum_dtype ............. None
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   gradient_accumulation_steps .. 2
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   gradient_clipping ............ 1.0
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   gradient_predivide_factor .... 1.0
[2024-10-22 20:05:09,906] [INFO] [config.py:1003:print]   graph_harvesting ............. False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   initial_dynamic_scale ........ 1
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   load_universal_checkpoint .... False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   loss_scale ................... 1.0
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   memory_breakdown ............. False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   mics_hierarchial_params_gather  False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   mics_shard_size .............. -1
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') comet=CometConfig(enabled=False, samples_log_interval=100, project=None, workspace=None, api_key=None, experiment_name=None, experiment_key=None, online=None, mode=None) wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName')
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   nebula_config ................ {
    "enabled": false,
    "persistent_storage_path": null,
    "persistent_time_interval": 100,
    "num_of_version_in_retention": 2,
    "enable_nebula_load": true,
    "load_path": null
}
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   optimizer_legacy_fusion ...... False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   optimizer_name ............... None
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   optimizer_params ............. None
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   pld_enabled .................. False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   pld_params ................... False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   prescale_gradients ........... False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   scheduler_name ............... None
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   scheduler_params ............. None
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   seq_parallel_communication_data_type  torch.float32
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   sparse_attention ............. None
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   sparse_gradients_enabled ..... False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   steps_per_print .............. inf
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   timers_config ................ enabled=True synchronized=True
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   train_batch_size ............. 2
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   train_micro_batch_size_per_gpu  1
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   use_data_before_expert_parallel_  False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   use_node_local_storage ....... False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   wall_clock_breakdown ......... False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   weight_quantization_config ... None
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   world_size ................... 1
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   zero_allow_untested_optimizer  True
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   zero_config .................. stage=0 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500000000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500000000 overlap_comm=True load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=None sub_group_size=1000000000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50000000 param_persistence_threshold=100000 model_persistence_threshold=9223372036854775807 max_live_parameters=1000000000 max_reuse_distance=1000000000 gather_16bit_weights_on_model_save=False use_all_reduce_for_fetch_params=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=True zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   zero_enabled ................. False
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   zero_force_ds_cpu_optimizer .. True
[2024-10-22 20:05:09,907] [INFO] [config.py:1003:print]   zero_optimization_stage ...... 0
[2024-10-22 20:05:09,908] [INFO] [config.py:989:print_user_config]   json = {
    "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 1.0,
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": false,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 0,
        "allgather_partitions": true,
        "allgather_bucket_size": 5.000000e+08,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5.000000e+08,
        "contiguous_gradients": true,
        "round_robin_gradients": true
    },
    "steps_per_print": inf
}
[INFO|trainer.py:2243] 2024-10-22 20:05:09,908 >> ***** Running training *****
[INFO|trainer.py:2244] 2024-10-22 20:05:09,908 >>   Num examples = 900
[INFO|trainer.py:2245] 2024-10-22 20:05:09,908 >>   Num Epochs = 3
[INFO|trainer.py:2246] 2024-10-22 20:05:09,908 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2249] 2024-10-22 20:05:09,908 >>   Total train batch size (w. parallel, distributed & accumulation) = 2
[INFO|trainer.py:2250] 2024-10-22 20:05:09,908 >>   Gradient Accumulation steps = 2
[INFO|trainer.py:2251] 2024-10-22 20:05:09,908 >>   Total optimization steps = 1,350
[INFO|trainer.py:2252] 2024-10-22 20:05:09,912 >>   Number of trainable parameters = 4,399,104
  0%|                                                                                                                                                                                                | 0/1350 [00:00<?, ?it/s]/home/hpc/anaconda3/envs/py310/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 3.7998, 'grad_norm': 3.6703357696533203, 'learning_rate': 7.4074074074074075e-06, 'epoch': 0.02}
{'loss': 3.7623, 'grad_norm': 4.231569290161133, 'learning_rate': 1.4814814814814815e-05, 'epoch': 0.04}
{'loss': 3.8555, 'grad_norm': 3.477386236190796, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.07}
{'loss': 3.4792, 'grad_norm': 2.956936836242676, 'learning_rate': 2.962962962962963e-05, 'epoch': 0.09}
{'loss': 3.6319, 'grad_norm': 3.6193692684173584, 'learning_rate': 3.7037037037037037e-05, 'epoch': 0.11}
{'loss': 3.5765, 'grad_norm': 3.526271343231201, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.13}
{'loss': 3.3857, 'grad_norm': 3.8371853828430176, 'learning_rate': 5.185185185185185e-05, 'epoch': 0.16}
{'loss': 3.5799, 'grad_norm': 3.627701759338379, 'learning_rate': 5.925925925925926e-05, 'epoch': 0.18}
{'loss': 3.3207, 'grad_norm': 3.9942593574523926, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.2}
{'loss': 3.5523, 'grad_norm': 4.094915390014648, 'learning_rate': 7.407407407407407e-05, 'epoch': 0.22}
{'loss': 3.426, 'grad_norm': 4.83339786529541, 'learning_rate': 8.148148148148148e-05, 'epoch': 0.24}
{'loss': 3.4403, 'grad_norm': 3.8962459564208984, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.27}
{'loss': 3.3958, 'grad_norm': 4.145336151123047, 'learning_rate': 9.62962962962963e-05, 'epoch': 0.29}
{'loss': 3.4668, 'grad_norm': 3.574202537536621, 'learning_rate': 9.999582149277187e-05, 'epoch': 0.31}
{'loss': 3.5005, 'grad_norm': 5.046082496643066, 'learning_rate': 9.996239762521151e-05, 'epoch': 0.33}
{'loss': 3.4016, 'grad_norm': 3.920306921005249, 'learning_rate': 9.989557223505661e-05, 'epoch': 0.36}
{'loss': 3.434, 'grad_norm': 4.520835876464844, 'learning_rate': 9.979538999730047e-05, 'epoch': 0.38}
{'loss': 3.4776, 'grad_norm': 4.332065105438232, 'learning_rate': 9.966191788709716e-05, 'epoch': 0.4}
{'loss': 3.4698, 'grad_norm': 4.012485027313232, 'learning_rate': 9.949524513498636e-05, 'epoch': 0.42}
{'loss': 3.5021, 'grad_norm': 3.976510763168335, 'learning_rate': 9.929548316723982e-05, 'epoch': 0.44}
{'loss': 3.4517, 'grad_norm': 3.442014455795288, 'learning_rate': 9.906276553136923e-05, 'epoch': 0.47}
{'loss': 3.3391, 'grad_norm': 3.136174440383911, 'learning_rate': 9.879724780684519e-05, 'epoch': 0.49}
{'loss': 3.2529, 'grad_norm': 3.762958288192749, 'learning_rate': 9.849910750108717e-05, 'epoch': 0.51}
{'loss': 3.4985, 'grad_norm': 5.989471912384033, 'learning_rate': 9.816854393079403e-05, 'epoch': 0.53}
{'loss': 3.396, 'grad_norm': 3.6840012073516846, 'learning_rate': 9.780577808869398e-05, 'epoch': 0.56}
{'loss': 3.2503, 'grad_norm': 3.8158295154571533, 'learning_rate': 9.741105249580383e-05, 'epoch': 0.58}
{'loss': 3.388, 'grad_norm': 3.782355785369873, 'learning_rate': 9.698463103929542e-05, 'epoch': 0.6}
{'loss': 3.3045, 'grad_norm': 3.439452648162842, 'learning_rate': 9.652679879607843e-05, 'epoch': 0.62}
{'loss': 3.4614, 'grad_norm': 3.6837027072906494, 'learning_rate': 9.603786184221693e-05, 'epoch': 0.64}
{'loss': 3.3844, 'grad_norm': 3.105792760848999, 'learning_rate': 9.551814704830734e-05, 'epoch': 0.67}
{'loss': 3.3395, 'grad_norm': 3.235607147216797, 'learning_rate': 9.496800186095466e-05, 'epoch': 0.69}
{'loss': 3.3238, 'grad_norm': 4.72987174987793, 'learning_rate': 9.438779407049281e-05, 'epoch': 0.71}
{'loss': 3.2436, 'grad_norm': 4.1176934242248535, 'learning_rate': 9.377791156510455e-05, 'epoch': 0.73}
{'loss': 3.3236, 'grad_norm': 3.872704267501831, 'learning_rate': 9.313876207150543e-05, 'epoch': 0.76}
{'loss': 3.1986, 'grad_norm': 3.317016363143921, 'learning_rate': 9.247077288236488e-05, 'epoch': 0.78}
{'loss': 3.2479, 'grad_norm': 3.488507032394409, 'learning_rate': 9.177439057064683e-05, 'epoch': 0.8}
{'loss': 3.2109, 'grad_norm': 3.3835136890411377, 'learning_rate': 9.105008069106093e-05, 'epoch': 0.82}
{'loss': 3.0966, 'grad_norm': 3.506141424179077, 'learning_rate': 9.029832746882371e-05, 'epoch': 0.84}
{'loss': 3.1959, 'grad_norm': 3.7045705318450928, 'learning_rate': 8.951963347593797e-05, 'epoch': 0.87}
{'loss': 3.2372, 'grad_norm': 5.036791801452637, 'learning_rate': 8.871451929520663e-05, 'epoch': 0.89}
{'loss': 3.1575, 'grad_norm': 3.091703176498413, 'learning_rate': 8.78835231722059e-05, 'epoch': 0.91}
{'loss': 3.2745, 'grad_norm': 3.7733335494995117, 'learning_rate': 8.702720065545024e-05, 'epoch': 0.93}
{'loss': 3.0976, 'grad_norm': 3.0565080642700195, 'learning_rate': 8.614612422498964e-05, 'epoch': 0.96}
{'loss': 3.3116, 'grad_norm': 5.735133647918701, 'learning_rate': 8.524088290968781e-05, 'epoch': 0.98}
{'loss': 3.2278, 'grad_norm': 3.519596576690674, 'learning_rate': 8.43120818934367e-05, 'epoch': 1.0}
{'loss': 3.1177, 'grad_norm': 3.403286933898926, 'learning_rate': 8.336034211057098e-05, 'epoch': 1.02}
{'loss': 3.2581, 'grad_norm': 3.8362677097320557, 'learning_rate': 8.238629983075294e-05, 'epoch': 1.04}
{'loss': 3.0941, 'grad_norm': 4.050426006317139, 'learning_rate': 8.139060623360493e-05, 'epoch': 1.07}
{'loss': 3.0806, 'grad_norm': 3.6774649620056152, 'learning_rate': 8.037392697337418e-05, 'epoch': 1.09}
{'loss': 2.9726, 'grad_norm': 3.909519672393799, 'learning_rate': 7.93369417339209e-05, 'epoch': 1.11}
 37%|███████████████████████████████████████████████████████████████████▍                                                                                                                  | 500/1350 [06:12<10:46,  1.31it/s][INFO|trainer.py:4021] 2024-10-22 20:11:22,693 >>
***** Running Evaluation *****
[INFO|trainer.py:4023] 2024-10-22 20:11:22,694 >>   Num examples = 100
[INFO|trainer.py:4026] 2024-10-22 20:11:22,694 >>   Batch size = 1
{'eval_loss': 3.297883987426758, 'eval_runtime': 4.3261, 'eval_samples_per_second': 23.115, 'eval_steps_per_second': 23.115, 'epoch': 1.11}
 37%|███████████████████████████████████████████████████████████████████▍                                                                                                                  | 500/1350 [06:17<10:46,  1.31it/s[INFO|trainer.py:3705] 2024-10-22 20:11:27,853 >> Saving model checkpoint to saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-500
[INFO|configuration_utils.py:673] 2024-10-22 20:11:27,868 >> loading configuration file /opt/xxx/models/Qwen2.5-0.5B-Instruct/config.json
[INFO|configuration_utils.py:742] 2024-10-22 20:11:27,868 >> Model config Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2641] 2024-10-22 20:11:27,888 >> tokenizer config file saved in saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-500/tokenizer_config.json
[INFO|tokenization_utils_base.py:2650] 2024-10-22 20:11:27,889 >> Special tokens file saved in saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-500/special_tokens_map.json
[2024-10-22 20:11:28,103] [INFO] [logging.py:96:log_dist] [Rank 0] [Torch] Checkpoint global_step500 is about to be saved!
[2024-10-22 20:11:28,124] [INFO] [logging.py:96:log_dist] [Rank 0] Saving model checkpoint: saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-500/global_step500/mp_rank_00_model_states.pt
[2024-10-22 20:11:28,124] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-500/global_step500/mp_rank_00_model_states.pt...
[2024-10-22 20:11:28,430] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-500/global_step500/mp_rank_00_model_states.pt.
[2024-10-22 20:11:28,431] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-500/global_step500/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt...
[2024-10-22 20:11:28,486] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-500/global_step500/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt.
[2024-10-22 20:11:28,486] [INFO] [engine.py:3536:_save_zero_checkpoint] bf16_zero checkpoint saved saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-500/global_step500/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
[2024-10-22 20:11:28,487] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step500 is ready now!
/home/hpc/anaconda3/envs/py310/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 3.0859, 'grad_norm': 3.913444757461548, 'learning_rate': 7.828034377432693e-05, 'epoch': 1.13}
{'loss': 3.0866, 'grad_norm': 3.976996421813965, 'learning_rate': 7.720483946542914e-05, 'epoch': 1.16}
{'loss': 3.19, 'grad_norm': 3.5531551837921143, 'learning_rate': 7.611114781758692e-05, 'epoch': 1.18}
{'loss': 3.0616, 'grad_norm': 4.464361190795898, 'learning_rate': 7.500000000000001e-05, 'epoch': 1.2}
{'loss': 3.0531, 'grad_norm': 4.418323040008545, 'learning_rate': 7.387213885189746e-05, 'epoch': 1.22}
{'loss': 3.036, 'grad_norm': 3.645994186401367, 'learning_rate': 7.272831838592503e-05, 'epoch': 1.24}
{'loss': 3.1175, 'grad_norm': 5.199030876159668, 'learning_rate': 7.156930328406268e-05, 'epoch': 1.27}
{'loss': 3.0264, 'grad_norm': 4.2354536056518555, 'learning_rate': 7.039586838640919e-05, 'epoch': 1.29}
{'loss': 3.0512, 'grad_norm': 4.233760833740234, 'learning_rate': 6.920879817317589e-05, 'epoch': 1.31}
 44%|███████████████████████████████████████████████████████████████████████████████▋                                                                                                                                          44%|███████████████████████████████████████████████████████████████████████████████▊                                                                                                                                          44%|███████████████████████████████████████████████████▊                                                                  | 593/1350 [07:27<09:22,  1.35it/s]                                                                {'loss': 3.3096, 'grad_norm': 5.272707939147949, 'learning_rate': 6.800888624023553e-05, 'epoch': 1.33}
{'loss': 3.0583, 'grad_norm': 4.071723937988281, 'learning_rate': 6.679693476857711e-05, 'epoch': 1.36}
{'loss': 2.9286, 'grad_norm': 3.8870277404785156, 'learning_rate': 6.557375398802123e-05, 'epoch': 1.38}
{'loss': 2.8992, 'grad_norm': 4.493603706359863, 'learning_rate': 6.434016163555452e-05, 'epoch': 1.4}
{'loss': 3.0966, 'grad_norm': 4.838465213775635, 'learning_rate': 6.30969824086453e-05, 'epoch': 1.42}
{'loss': 3.1561, 'grad_norm': 4.321047306060791, 'learning_rate': 6.184504741390596e-05, 'epoch': 1.44}
{'loss': 2.9638, 'grad_norm': 4.193413257598877, 'learning_rate': 6.058519361147055e-05, 'epoch': 1.47}
{'loss': 3.0097, 'grad_norm': 4.506619453430176, 'learning_rate': 5.9318263255459116e-05, 'epoch': 1.49}
{'loss': 3.0276, 'grad_norm': 3.7007498741149902, 'learning_rate': 5.804510333090287e-05, 'epoch': 1.51}
{'loss': 3.0741, 'grad_norm': 4.983395576477051, 'learning_rate': 5.6766564987506566e-05, 'epoch': 1.53}
{'loss': 2.9993, 'grad_norm': 4.620068550109863, 'learning_rate': 5.548350297062659e-05, 'epoch': 1.56}
{'loss': 3.0054, 'grad_norm': 4.520689487457275, 'learning_rate': 5.419677504984534e-05, 'epoch': 1.58}
{'loss': 3.0005, 'grad_norm': 5.8599138259887695, 'learning_rate': 5.290724144552379e-05, 'epoch': 1.6}
{'loss': 3.1859, 'grad_norm': 3.9076790809631348, 'learning_rate': 5.1615764253715536e-05, 'epoch': 1.62}
{'loss': 3.0262, 'grad_norm': 5.371761798858643, 'learning_rate': 5.0323206869826966e-05, 'epoch': 1.64}
{'loss': 3.1393, 'grad_norm': 4.425382137298584, 'learning_rate': 4.903043341140879e-05, 'epoch': 1.67}
{'loss': 2.9784, 'grad_norm': 4.30043888092041, 'learning_rate': 4.7738308140464685e-05, 'epoch': 1.69}
{'loss': 3.001, 'grad_norm': 5.003708839416504, 'learning_rate': 4.6447694885663514e-05, 'epoch': 1.71}
{'loss': 2.9195, 'grad_norm': 4.5432000160217285, 'learning_rate': 4.515945646484105e-05, 'epoch': 1.73}
{'loss': 2.8774, 'grad_norm': 4.6029953956604, 'learning_rate': 4.387445410817774e-05, 'epoch': 1.76}
{'loss': 3.0632, 'grad_norm': 5.247580528259277, 'learning_rate': 4.259354688243757e-05, 'epoch': 1.78}
{'loss': 3.0326, 'grad_norm': 4.038956642150879, 'learning_rate': 4.131759111665349e-05, 'epoch': 1.8}
{'loss': 3.0326, 'grad_norm': 4.653959274291992, 'learning_rate': 4.004743982964298e-05, 'epoch': 1.82}
{'loss': 2.9738, 'grad_norm': 4.61118221282959, 'learning_rate': 3.878394215973663e-05, 'epoch': 1.84}
{'loss': 3.15, 'grad_norm': 4.504194736480713, 'learning_rate': 3.752794279710094e-05, 'epoch': 1.87}
{'loss': 3.0874, 'grad_norm': 5.148902893066406, 'learning_rate': 3.628028141903493e-05, 'epoch': 1.89}
{'loss': 3.0156, 'grad_norm': 4.611048698425293, 'learning_rate': 3.5041792128617927e-05, 'epoch': 1.91}
{'loss': 2.9946, 'grad_norm': 4.30831241607666, 'learning_rate': 3.381330289708396e-05, 'epoch': 1.93}
{'loss': 3.1274, 'grad_norm': 4.7987284660339355, 'learning_rate': 3.2595635010295475e-05, 'epoch': 1.96}
{'loss': 3.1876, 'grad_norm': 5.304200649261475, 'learning_rate': 3.1389602519686515e-05, 'epoch': 1.98}
{'loss': 3.0889, 'grad_norm': 4.293857097625732, 'learning_rate': 3.019601169804216e-05, 'epoch': 2.0}
{'loss': 2.7142, 'grad_norm': 4.179135799407959, 'learning_rate': 2.901566050047855e-05, 'epoch': 2.02}
{'loss': 3.08, 'grad_norm': 5.14447021484375, 'learning_rate': 2.7849338030983257e-05, 'epoch': 2.04}
{'loss': 2.8977, 'grad_norm': 4.8162007331848145, 'learning_rate': 2.6697824014873075e-05, 'epoch': 2.07}
{'loss': 3.0984, 'grad_norm': 4.361102104187012, 'learning_rate': 2.5561888277521794e-05, 'epoch': 2.09}
{'loss': 2.9275, 'grad_norm': 4.269589900970459, 'learning_rate': 2.4442290229706344e-05, 'epoch': 2.11}
{'loss': 2.8893, 'grad_norm': 4.506771564483643, 'learning_rate': 2.333977835991545e-05, 'epoch': 2.13}
{'loss': 2.8044, 'grad_norm': 4.541017055511475, 'learning_rate': 2.225508973396016e-05, 'epoch': 2.16}
{'loss': 2.8541, 'grad_norm': 4.707365036010742, 'learning_rate': 2.1188949502220983e-05, 'epoch': 2.18}
{'loss': 2.7467, 'grad_norm': 4.4472246170043945, 'learning_rate': 2.0142070414860704e-05, 'epoch': 2.2}
{'loss': 2.8241, 'grad_norm': 4.715089321136475, 'learning_rate': 1.9115152345327152e-05, 'epoch': 2.22}
 74%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                           | 1000/1350 [12:29<04:27,  1.31it/s]              [INFO|trainer.py:4021] 2024-10-22 20:17:39,603 >>
***** Running Evaluation *****
[INFO|trainer.py:4023] 2024-10-22 20:17:39,603 >>   Num examples = 100
[INFO|trainer.py:4026] 2024-10-22 20:17:39,603 >>   Batch size = 1
{'eval_loss': 3.2544291019439697, 'eval_runtime': 4.1174, 'eval_samples_per_second': 24.287, 'eval_steps_per_second': 24.287, 'epoch': 2.22}
 74%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                           | 1000/1350 [12:33<04:27,  1.31it/s[              INFO|trainer.py:3705] 2024-10-22 20:17:44,372 >> Saving model checkpoint to saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1000
[INFO|configuration_utils.py:673] 2024-10-22 20:17:44,385 >> loading configuration file /opt/xxx/models/Qwen2.5-0.5B-Instruct/config.json
[INFO|configuration_utils.py:742] 2024-10-22 20:17:44,386 >> Model config Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2641] 2024-10-22 20:17:44,406 >> tokenizer config file saved in saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1000/tokenizer_config.json
[INFO|tokenization_utils_base.py:2650] 2024-10-22 20:17:44,406 >> Special tokens file saved in saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1000/special_tokens_map.json
[2024-10-22 20:17:44,598] [INFO] [logging.py:96:log_dist] [Rank 0] [Torch] Checkpoint global_step1000 is about to be saved!
[2024-10-22 20:17:44,618] [INFO] [logging.py:96:log_dist] [Rank 0] Saving model checkpoint: saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1000/global_step1000/mp_rank_00_model_states.pt
[2024-10-22 20:17:44,618] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1000/global_step1000/mp_rank_00_model_states.pt...
[2024-10-22 20:17:44,933] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1000/global_step1000/mp_rank_00_model_states.pt.
[2024-10-22 20:17:44,934] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1000/global_step1000/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt...
[2024-10-22 20:17:44,990] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1000/global_step1000/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt.
[2024-10-22 20:17:44,991] [INFO] [engine.py:3536:_save_zero_checkpoint] bf16_zero checkpoint saved saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1000/global_step1000/bf16_zero_pp_rank_0_mp_rank_00_optim_sta              tes.pt
[2024-10-22 20:17:44,991] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step1000 is ready now!
/home/hpc/anaconda3/envs/py310/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` inste              ad.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 2.7679, 'grad_norm': 5.567869186401367, 'learning_rate': 1.8108881822464696e-05, 'epoch': 2.24}
{'loss': 2.8003, 'grad_norm': 4.951780796051025, 'learning_rate': 1.7123931571546827e-05, 'epoch': 2.27}
{'loss': 2.7721, 'grad_norm': 6.794727802276611, 'learning_rate': 1.6160960064536908e-05, 'epoch': 2.29}
{'loss': 2.6824, 'grad_norm': 5.510177135467529, 'learning_rate': 1.52206110798779e-05, 'epoch': 2.31}
{'loss': 2.9909, 'grad_norm': 4.82781982421875, 'learning_rate': 1.4303513272105057e-05, 'epoch': 2.33}
{'loss': 2.708, 'grad_norm': 6.547961711883545, 'learning_rate': 1.3410279751569399e-05, 'epoch': 2.36}
{'loss': 2.8044, 'grad_norm': 5.56920862197876, 'learning_rate': 1.25415076745532e-05, 'epoch': 2.38}
{'loss': 2.8533, 'grad_norm': 6.193923473358154, 'learning_rate': 1.1697777844051105e-05, 'epoch': 2.4}
{'loss': 2.7354, 'grad_norm': 4.634361743927002, 'learning_rate': 1.0879654321484012e-05, 'epoch': 2.42}
{'loss': 2.7282, 'grad_norm': 5.485413551330566, 'learning_rate': 1.008768404960535e-05, 'epoch': 2.44}
{'loss': 2.6121, 'grad_norm': 6.07513952255249, 'learning_rate': 9.322396486851626e-06, 'epoch': 2.47}
{'loss': 2.8807, 'grad_norm': 5.442063808441162, 'learning_rate': 8.584303253381847e-06, 'epoch': 2.49}
{'loss': 2.8014, 'grad_norm': 5.05751371383667, 'learning_rate': 7.873897789042523e-06, 'epoch': 2.51}
{'loss': 2.8563, 'grad_norm': 5.755252361297607, 'learning_rate': 7.191655023486682e-06, 'epoch': 2.53}
{'loss': 2.828, 'grad_norm': 5.505600452423096, 'learning_rate': 6.53803105866761e-06, 'epoch': 2.56}
{'loss': 2.7639, 'grad_norm': 6.044736862182617, 'learning_rate': 5.9134628639196e-06, 'epoch': 2.58}
{'loss': 2.7516, 'grad_norm': 4.429302215576172, 'learning_rate': 5.318367983829392e-06, 'epoch': 2.6}
{'loss': 2.7925, 'grad_norm': 5.318936824798584, 'learning_rate': 4.7531442590937335e-06, 'epoch': 2.62}
{'loss': 2.8191, 'grad_norm': 5.769840240478516, 'learning_rate': 4.218169560549706e-06, 'epoch': 2.64}
{'loss': 2.8287, 'grad_norm': 5.4126787185668945, 'learning_rate': 3.7138015365554833e-06, 'epoch': 2.67}
{'loss': 2.9877, 'grad_norm': 5.435266971588135, 'learning_rate': 3.2403773738905187e-06, 'epoch': 2.69}
{'loss': 2.7879, 'grad_norm': 6.221948146820068, 'learning_rate': 2.798213572335001e-06, 'epoch': 2.71}
{'loss': 2.9761, 'grad_norm': 5.835367679595947, 'learning_rate': 2.3876057330792346e-06, 'epoch': 2.73}
{'loss': 2.6658, 'grad_norm': 4.397183895111084, 'learning_rate': 2.0088283611044036e-06, 'epoch': 2.76}
{'loss': 3.006, 'grad_norm': 5.709335803985596, 'learning_rate': 1.6621346816668992e-06, 'epoch': 2.78}
{'loss': 2.8045, 'grad_norm': 6.681838512420654, 'learning_rate': 1.3477564710088098e-06, 'epoch': 2.8}
{'loss': 2.7259, 'grad_norm': 5.518866539001465, 'learning_rate': 1.0659039014077944e-06, 'epoch': 2.82}
{'loss': 2.9063, 'grad_norm': 5.409673690795898, 'learning_rate': 8.167654006699443e-07, 'epoch': 2.84}
{'loss': 2.8843, 'grad_norm': 4.7312822341918945, 'learning_rate': 6.005075261595494e-07, 'epoch': 2.87}
{'loss': 2.9639, 'grad_norm': 5.868640899658203, 'learning_rate': 4.1727485344994486e-07, 'epoch': 2.89}
{'loss': 2.796, 'grad_norm': 6.320312023162842, 'learning_rate': 2.671898796699268e-07, 'epoch': 2.91}
{'loss': 2.8348, 'grad_norm': 6.177616596221924, 'learning_rate': 1.503529416103988e-07, 'epoch': 2.93}
{'loss': 2.8771, 'grad_norm': 4.994110107421875, 'learning_rate': 6.684214864584038e-08, 'epoch': 2.96}
{'loss': 3.0045, 'grad_norm': 5.709598541259766, 'learning_rate': 1.6713330515627513e-08, 'epoch': 2.98}
{'loss': 2.9264, 'grad_norm': 5.61161994934082, 'learning_rate': 0.0, 'epoch': 3.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1350/1350 [16:59<00:00,  1.30it/s]              [INFO|trainer.py:3705] 2024-10-22 20:22:10,539 >> Saving model checkpoint to saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1350
[INFO|configuration_utils.py:673] 2024-10-22 20:22:10,562 >> loading configuration file /opt/xxx/models/Qwen2.5-0.5B-Instruct/config.json
[INFO|configuration_utils.py:742] 2024-10-22 20:22:10,563 >> Model config Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2641] 2024-10-22 20:22:10,592 >> tokenizer config file saved in saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1350/tokenizer_config.json
[INFO|tokenization_utils_base.py:2650] 2024-10-22 20:22:10,592 >> Special tokens file saved in saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1350/special_tokens_map.json
[2024-10-22 20:22:10,787] [INFO] [logging.py:96:log_dist] [Rank 0] [Torch] Checkpoint global_step1350 is about to be saved!
[2024-10-22 20:22:10,807] [INFO] [logging.py:96:log_dist] [Rank 0] Saving model checkpoint: saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1350/global_step1350/mp_rank_00_model_states.pt
[2024-10-22 20:22:10,807] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1350/global_step1350/mp_rank_00_model_states.pt...
[2024-10-22 20:22:11,118] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1350/global_step1350/mp_rank_00_model_states.pt.
[2024-10-22 20:22:11,119] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1350/global_step1350/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt...
[2024-10-22 20:22:11,173] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1350/global_step1350/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt.
[2024-10-22 20:22:11,174] [INFO] [engine.py:3536:_save_zero_checkpoint] bf16_zero checkpoint saved saves/Qwen2.5-0.5B-Instruct/lora/sft/checkpoint-1350/global_step1350/bf16_zero_pp_rank_0_mp_rank_00_optim_sta              tes.pt
[2024-10-22 20:22:11,174] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step1350 is ready now!
[INFO|trainer.py:2505] 2024-10-22 20:22:11,178 >>

Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 1021.2666, 'train_samples_per_second': 2.644, 'train_steps_per_second': 1.322, 'train_loss': 3.096851877283167, 'epoch': 3.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1350/1350 [17:01<00:00,  1.32it/s]
[INFO|trainer.py:3705] 2024-10-22 20:22:11,581 >> Saving model checkpoint to saves/Qwen2.5-0.5B-Instruct/lora/sft
[INFO|configuration_utils.py:673] 2024-10-22 20:22:11,594 >> loading configuration file /opt/xxx/models/Qwen2.5-0.5B-Instruct/config.json
[INFO|configuration_utils.py:742] 2024-10-22 20:22:11,594 >> Model config Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

[INFO|tokenization_utils_base.py:2641] 2024-10-22 20:22:11,614 >> tokenizer config file saved in saves/Qwen2.5-0.5B-Instruct/lora/sft/tokenizer_config.json
[INFO|tokenization_utils_base.py:2650] 2024-10-22 20:22:11,614 >> Special tokens file saved in saves/Qwen2.5-0.5B-Instruct/lora/sft/special_tokens_map.json
***** train metrics *****
  epoch                    =        3.0
  total_flos               =   766230GF
  train_loss               =     3.0969
  train_runtime            = 0:17:01.26
  train_samples_per_second =      2.644
  train_steps_per_second   =      1.322
Figure saved at: saves/Qwen2.5-0.5B-Instruct/lora/sft/training_loss.png
Figure saved at: saves/Qwen2.5-0.5B-Instruct/lora/sft/training_eval_loss.png
10/22/2024 20:22:12 - WARNING - llamafactory.extras.ploting - No metric eval_accuracy to plot.
[INFO|trainer.py:4021] 2024-10-22 20:22:12,031 >>
***** Running Evaluation *****
[INFO|trainer.py:4023] 2024-10-22 20:22:12,031 >>   Num examples = 100
[INFO|trainer.py:4026] 2024-10-22 20:22:12,031 >>   Batch size = 1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:04<00:00, 24.27it/s]
***** eval metrics *****
  epoch                   =        3.0
  eval_loss               =      3.254
  eval_runtime            = 0:00:04.17
  eval_samples_per_second =     23.978
  eval_steps_per_second   =     23.978
[INFO|modelcard.py:449] 2024-10-22 20:22:16,204 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}
(py310) [root@hpc-009 LLaMA-Factory]# ls saves/Qwen2.5-0.5B-Instruct/lora/sft/
adapter_config.json        added_tokens.json  checkpoint-1000  checkpoint-500     merges.txt  runs                     tokenizer_config.json  trainer_log.jsonl   training_args.bin       training_loss.png   vocab.json
adapter_model.safetensors  all_results.json   checkpoint-1350  eval_results.json  README.md   special_tokens_map.json  tokenizer.json         trainer_state.json  training_eval_loss.png  train_results.json
(py310) [root@hpc-009 LLaMA-Factory]# ls saves/Qwen2.5-0.5B-Instruct/lora/sft/ -l
total 24348
-rw-r--r-- 1 root root      743 Oct 22 20:22 adapter_config.json
-rw-r--r-- 1 root root  8841928 Oct 22 20:22 adapter_model.safetensors
-rw-r--r-- 1 root root      605 Oct 22 20:22 added_tokens.json
-rw-r--r-- 1 root root      343 Oct 22 20:22 all_results.json
drwxr-xr-x 3 root root     4096 Oct 22 20:17 checkpoint-1000
drwxr-xr-x 3 root root     4096 Oct 22 20:22 checkpoint-1350
drwxr-xr-x 3 root root     4096 Oct 22 20:11 checkpoint-500
-rw-r--r-- 1 root root      161 Oct 22 20:22 eval_results.json
-rw-r--r-- 1 root root  1671853 Oct 22 20:22 merges.txt
-rw-r--r-- 1 root root     1568 Oct 22 20:22 README.md
drwxr-xr-x 4 root root     4096 Oct 22 20:05 runs
-rw-r--r-- 1 root root      613 Oct 22 20:22 special_tokens_map.json
-rw-r--r-- 1 root root     7333 Oct 22 20:22 tokenizer_config.json
-rw-r--r-- 1 root root 11421896 Oct 22 20:22 tokenizer.json
-rw-r--r-- 1 root root    28077 Oct 22 20:22 trainer_log.jsonl
-rw-r--r-- 1 root root    24442 Oct 22 20:22 trainer_state.json
-rw-r--r-- 1 root root     6648 Oct 22 20:22 training_args.bin
-rw-r--r-- 1 root root    38524 Oct 22 20:22 training_eval_loss.png
-rw-r--r-- 1 root root    52632 Oct 22 20:22 training_loss.png
-rw-r--r-- 1 root root      202 Oct 22 20:22 train_results.json
-rw-r--r-- 1 root root  2776833 Oct 22 20:22 vocab.json
```
权重合并
```python
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
from peft import LoraConfig, AutoPeftModelForCausalLM
 
 
# 2. load PEFT model in fp16
model = AutoPeftModelForCausalLM.from_pretrained(
    'saves/Qwen2.5-0.5B-Instruct/lora/sft',
    device_map="auto",
    torch_dtype=torch.float16,
    # torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
 
# 3. merge adapter weights with base model
merged_model = model.merge_and_unload()
 
merged_model.save_pretrained('saves/Qwen2.5-0.5B-Instruct/lora/sft/merged', safe_serialization=True, max_shard_size="2GB")
```
合并后的权重：
```python
(py310) [root@hpc-009 merged]# ls -l
total 975704
-rw-r--r-- 1 root root       746 Oct 23 09:28 config.json
-rw-r--r-- 1 root root       242 Oct 23 09:28 generation_config.json
-rw-r--r-- 1 root root   1671839 Oct 23 09:33 merges.txt
-rw-r--r-- 1 root root 987611904 Oct 23 09:28 model.safetensors
-rw-r--r-- 1 root root      7305 Oct 23 09:33 tokenizer_config.json
-rw-r--r-- 1 root root   7031645 Oct 23 09:33 tokenizer.json
-rw-r--r-- 1 root root   2776833 Oct 23 09:33 vocab.json
```
##4.2 推理测试
使用vLLM对模型进行加载并运行为类OpenAI API服务，即可通过API调用进行推理测试。
```python
#!/bin/bash


export CUDA_VISIBLE_DEVICES="0"

vllm serve saves/Qwen2.5-0.5B-Instruct/lora/sft/merged --port 26942
```
测试代码：
```python
import openai

client = openai.OpenAI(
    base_url="http://hpc-009:26942/v1",
    api_key = "sk-no-key-required"
)

def sendChat(messages):
    response = client.chat.completions.create(
        model='saves/Qwen2.5-0.5B-Instruct/lora/sft/merged',
        temperature=0.01,
        messages=messages, stop="<|im_end|>",
        stream=True
    )

    from IPython.display import display, HTML
    cur = ''
    for r in response:
        try:
            if r.choices[0].delta.content is not None:
                cur += r.choices[0].delta.content
                display(HTML(cur), clear=True)
        except Exception as e:
            print(e)
            continue
    
    return cur

messages2 = [{'role': 'system',
  'content': 'You are a helpful assistant.'},
 {'role': 'user',
  'content': '类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤'}]

response = sendChat(messages2)
```
输出：这款阔腿裤的版型设计，宽松的裤腿线条，可以很好的修饰腿部的线条，让双腿看起来更加修长。而裤身的两侧，分别设计了性感的荷叶边，让整体的造型更加的浪漫。
测试效果截图：
对比微调前的大模型输出效果如下：
