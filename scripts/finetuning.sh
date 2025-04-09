cd /home/xh/memVP
#if use llama2 the normal_scale should be 0.2 to avoid gradient explode
#if use llama1 the normal_scale can be 1
""" anyway,if you want to change the top_genes & emb,
 you may need to change the normal_scale and adapter_scale due to the activate function in adapter is silu not softmax """

CUDA_VISIBLE_DEVICES=1,0,2,3 torchrun --nproc_per_node 4 --master_port 13345 train.py \
    --llama_model_path /HDDDATA/XieeeHuiii/checkpoint/llama-2-7b-chat1/ \
    --data_path /HDDDATA/XieeeHuiii/Data/data/immune/train_data.h5ad \
    --model_file /home/xh/memVP/scgpt/checkpoint/scgpt-human/best_model.pt \
    --config_file ./scgpt/checkpoint/scgpt-human/args.json \
    --vocab_file ./scgpt/checkpoint/scgpt-human/vocab.json \
    --max_seq_len 512 \
    --batch_size 1 \
    --accum_iter 4 \
    --epochs 20 \
    --warmup_epochs 1 \
    --blr 2e-3 \
    --weight_decay 0.02 \
    --output_dir ./checkpoint/MemVP-immune-4096/ \
    --adapter_dim 12 \
    --adapter_scale 0.02 \
    --normal_scale 0.2 \
    --prompt_format QCM-A \
    --select_label Manually_curated_celltype \
    --seed 88 --emb 4096 --top_genes 4096
