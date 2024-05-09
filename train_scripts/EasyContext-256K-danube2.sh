# First, download h2oai/h2o-danube2-1.8b-base to output/h2o-danube2-1.8b-base and change the "architectures" from MistralForCausalLM to LlamaForCausalLM and "model_type" from mistral to llama in the config.json file.
# HF's mistral architecture is off when used together with ring attention due to the way it implements rotary embedding. Llama has no such issue.


accelerate launch --config_file accelerate_configs/single_node.yaml --main_process_port 12345 train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o \
--wandb EasyContext \
--seed 2026 \
--max-train-steps 400 \
--learning-rate 2e-5 \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model output \
--seq-length 32000 \
--rope-theta 300000 \
--parallel_mode data_parallel






