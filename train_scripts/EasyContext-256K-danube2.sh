# First, download h2oai/h2o-danube2-1.8b-base to output/h2o-danube2-1.8b-base and change the "architectures" from MistralForCausalLM to LlamaForCausalLM and "model_type" from mistral to llama in the config.json file.
# HF's mistral architecture is off when used together with ring attention due to the way it implements rotary embedding. Llama has no such issue.


accelerate launch --config_file accelerate_configs/single_node.yaml --main_process_port 12345 train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o \
#--wandb EasyContext \
--seed 2026 \
--max-train-steps 400 \
--learning-rate 2e-5 \
--dataset damerajee/tokenized_long_context_hin \
--model damerajee/Tiny-OpenHathi-v4 \
--seq-length 32000 \
--rope-theta 300000 \
--parallel_mode data_parallel






