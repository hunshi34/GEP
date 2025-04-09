cd /home/xh/memVP
CUDA_VISIBLE_DEVICES=2 python eval_1.py \
    --model 7B \
    --adapter_path /home/xh/memVP/checkpoint/MemVP-immune-nogene/  \
    --data_path /HDDDATA/XieeeHuiii/Data/data/immune/test_data.h5ad  \
    --model_file /home/xh/memVP/checkpoint/MemVP-immune-nogene/backbone_epoch_19.pt \
    --gene_list /home/xh/memVP/checkpoint/MemVP-immune-nogene/select_genes.json \

