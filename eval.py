from GEP.build import create_model
import torch
import argparse
from dataclasses import dataclass
from GEP.tokenizer import Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
import pandas as pd
import torch.utils.data as Data
from tqdm import tqdm
import re
from util.datasets import (hPancreasDataSet, MyeloidDataSet,msDataSet,msDataSet2,simulationDataset,
                           hanpdataset,heartdataset,Macrophagesdataset,cancerdataset,immunedataset)

@dataclass
class PromptArgs:
    prompt_format = 'QCM-A'
    use_caption = True
    options = ["A", "B", "C", "D", "E"]


def extract_label(pred):
    match = re.search(r'\d+', pred)
    if match:
        return int(match.group(0))
    return None

def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1  # return random.choice(range(len(choices)))


@dataclass
class ModelArgs_7B:
    # llama_model_path = '/home/xh/memVP/weight/llama/'
    llama_model_path ="/HDDDATA/XieeeHuiii/checkpoint/llama-2-7b-chat/"
    llm_model = '7B'
    max_seq_len = 768
    hidden_proj = 128
    emb = 1024
    cpu_load = False
    adapter_scale = 0.02
    normal_scale =0.2
    adapter_dim = 12
    max_batch_size=4
    gradient_checkpointing = True
    is_train = True
    data_path= '/HDDDATA/XieeeHuiii/simulation/simulation_rna.h5ad'
    model_file= '/home/xh/memVP/scgpt/checkpoint/scgpt-human/best_model.pt'
    config_file= '/home/xh/memVP/scgpt/checkpoint/scgpt-human/args.json'
    vocab_file ='/home/xh/memVP/scgpt/checkpoint/scgpt-human/vocab.json'
    use_batch_labels=0

@dataclass
class ModelArgs_13B:
    llama_model_path = './data/weights/'
    llm_model = '13B'
    max_seq_len = 512
    hidden_proj = 128
    emb = 400
    cpu_load = False
    adapter_scale = 0.1
    adapter_dim = 12
    gradient_checkpointing = False
    is_train = False
    data_root = './data/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--model', default='7B', type=str)
    parser.add_argument('--adapter_path', default='MemVP-SQA-7B', type=str)
    parser.add_argument('--gene_list', default=None, type=str)
    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--model_file', default='/home/xh/memVP/scgpt/checkpoint/scgpt-human/best_model.pt', type=str)
    parser.add_argument('--config_file', default=None, type=str)
    parser.add_argument('--vocab_file', default=None, type=str)
    parser.add_argument('--llama_model_path', default=None, type=str)

    args = parser.parse_args()
    config_file=args.config_file
    vocab_file=args.vocab_file
    llama_model_path=args.llama_model_path
    bs = args.batch_size
    adapter_path = args.adapter_path
    data_path = args.data_path
    model_file = args.model_file
    with open(args.gene_list, 'r') as f:
        gene_list = json.load(f)
    if args.model == '7B':
        args = ModelArgs_7B()
    else:
        args = ModelArgs_13B()
    if config_file is not None:
        args.config_file = config_file
    if vocab_file is not None:
        args.vocab_file = vocab_file

    if llama_model_path is not None:
        args.llama_model_path = llama_model_path
    args.data_path=data_path
    args.model_file=model_file
    if "heart" in data_path:
        test_data =heartdataset(args, args.llama_model_path,top_genes=args.emb, is_eval=True, top_gene_list=gene_list,select_label="cell_type_leiden0.6")
    elif "Macrophages" in data_path:
        test_data = Macrophagesdataset(args, args.llama_model_path, top_genes=args.emb,is_eval=True, top_gene_list=gene_list, select_label='new_Cell_Type')
    elif "cancer" in args.data_path:
        test_data =cancerdataset(args, args.llama_model_path, top_genes=args.emb,is_eval=True, top_gene_list=gene_list,select_label="CellType")
    elif "immune" in args.data_path:
        test_data=immunedataset(args, args.llama_model_path, top_genes=args.emb,is_eval=True, top_gene_list=gene_list,select_label="Manually_curated_celltype")
    else:
        test_data =hanpdataset(args, args.llama_model_path,top_genes=args.emb, is_eval=True, top_gene_list=gene_list,select_label="full_clustering")

    args.label_length=test_data.label_length
    llama = create_model(args)
    llama.half()
    device = torch.device("cuda")
    llama.to(device)
    adapter = torch.load(os.path.join(adapter_path, 'checkpoint-19.pth'))['model']
    sd = {}
    for k in adapter:
        sd[k.replace('module.', '')] = adapter[k]
    llama.load_state_dict(sd, False)
    classifier_path=os.path.join(adapter_path, 'classifier_epoch_19.pt')
    llama.classifier.load_state_dict(torch.load(classifier_path,map_location=device), False)
    # llama.classifier.to(device)
    tokenizer = Tokenizer(model_path=os.path.join(args.llama_model_path, 'tokenizer.model'))

    dataloader = Data.DataLoader(test_data, batch_size=bs)
    preds=[]
    s,h=0,0
    for prompts, label,example_mask,genes, values, gene_mask,*batch in tqdm(dataloader):
        if batch:
            batch=batch[0]
        else:
            batch=None
        prompts = prompts.to(device)
        genes = genes.to(device)
        values = values.to(device)
        gene_mask = gene_mask.to(device)
        # print(genes ,values,gene_mask)
        with torch.inference_mode():
            results = llama(prompts, None,src=genes,value=values,src_key_padding_mask=gene_mask, batch=batch)
        if type(label)==tuple:
            label = list(label)
        # print(results.shape)
        for result,i1 in zip(results,label):
            predicted_label = torch.argmax(result, dim=-1)
            # 将 predicted_label 和 i1 转换为 numpy 数组或 Python 列表
            predicted_label = predicted_label.cpu().detach().item() # 如果使用GPU
            i1 = i1.cpu().detach().item()  if isinstance(i1, torch.Tensor) else i1  # 如果i1是tensor

            preds.append({"pred": predicted_label, "label": i1})
    mean_attention_means=llama.cumulative_attention_means
    mean_attention_means_json = {
        layer_idx: values.cpu().numpy().tolist()  # 转换为 NumPy 数组，再转换为列表
        for layer_idx, values in mean_attention_means.items()
    }
    output=llama.output_hidden
    output=[i.tolist() for i in output]
    with open(os.path.join(adapter_path,'attention_output.json'), 'w') as f:
        json.dump(output, f, indent=4)  # indent 用于美化输出
    # 保存为 JSON 文件
    with open(os.path.join(adapter_path,'mean_attention_means.json'), 'w') as f:
        json.dump(mean_attention_means_json, f, indent=4)  # indent 用于美化输出
        # 将所有预测和标签收集到列表中


    with open(os.path.join(adapter_path, "preds9.json"), 'w') as f:
        json.dump(preds, f, indent=4)
    predictions = [p["pred"] for p in preds]
    labels = [p["label"] for p in preds]
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro') # 可以根据需要调整为 'micro', 'weighted'
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

if __name__ == '__main__':
    main()
