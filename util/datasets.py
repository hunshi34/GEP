import json, random
import torch.utils.data as Data
import os
import pandas as pd
import torch
from GEP import Tokenizer
import copy
import scanpy as sc
import numpy as np
from scgpt.tokenizer import  GeneVocab
# global_cell_types = ['acinar', 'delta', 'beta', 'PSC', 'alpha', 'ductal', 'epsilon', 'PP', 'endothelial',
#                      'macrophage', 'schwann', 'mast', 't_cell', 'MHC class II']

##MS
# global_cell_types = ['PVALB-expressing interneuron', 'SST-expressing interneuron',
#                      'SV2C-expressing interneuron', 'VIP-expressing interneuron', 'astrocyte',
#                      'cortical layer 2-3 excitatory neuron A', 'cortical layer 2-3 excitatory neuron B',
#                      'cortical layer 4 excitatory neuron', 'cortical layer 5-6 excitatory neuron',
#                      'endothelial cell', 'microglial cell', 'mixed excitatory neuron',
#                      'mixed glial cell?', 'oligodendrocyte A', 'oligodendrocyte C',
#                      'oligodendrocyte precursor cell', 'phagocyte', 'pyramidal neuron?']
# select_data = ["celltype", "Sample Characteristic[sampling site]", "Sample Characteristic[organism part]",
#                    "Sample Characteristic[age]", "Sample Characteristic[sex]"]

###simulation
# global_cell_types =['CD4 CTL', 'CD4 Naive', 'CD4 Proliferating', 'CD4 TCM_1', 'CD4 TCM_2',
#        'CD4 TCM_3', 'CD4 TEM_1', 'CD4 TEM_2', 'CD4 TEM_3', 'CD4 TEM_4',
#        'CD8 Naive', 'CD8 Naive_2', 'CD8 Proliferating', 'CD8 TCM_1',
#        'CD8 TCM_2', 'CD8 TCM_3', 'CD8 TEM_1', 'CD8 TEM_2', 'CD8 TEM_3',
#        'CD8 TEM_4', 'CD8 TEM_5', 'CD8 TEM_6', 'CD14 Mono', 'CD16 Mono',
#        'NK Proliferating', 'NK_1', 'NK_2', 'NK_3', 'NK_4', 'NK_CD56bright',
#        'Treg Memory', 'Treg Naive']
# global_cell_types =['CD14 Mono', 'CD4 TCM_1', 'CD8 Naive', 'NK_2', 'CD8 TEM_1',
#                     'CD16 Mono', 'B intermediate lambda', 'CD4 Naive', 'CD4 CTL', 'B naive kappa',
#                     'CD4 TCM_3', 'MAIT', 'CD4 TCM_2', 'CD8 TEM_2', 'gdT_3', 'NK_1', 'CD8 TCM_1', 'dnT_2',
#                     'B intermediate kappa', 'B memory kappa', 'Doublet', 'pDC', 'CD8 TEM_5', 'gdT_1', 'B naive lambda',
#                     'NK_4', 'CD8 Proliferating', 'CD8 TCM_2', 'Treg Naive', 'Plasma', 'CD4 TEM_1', 'Treg Memory',
#                     'CD4 TEM_3', 'CD8 TCM_3', 'cDC2_1', 'NK Proliferating', 'CD8 TEM_4', 'ASDC_pDC', 'CD4 TEM_2',
#                     'B memory lambda', 'dnT_1', 'HSPC', 'cDC2_2', 'Platelet', 'NK_CD56bright', 'CD4 TEM_4', 'CD8 TEM_6',
#                     'CD8 Naive_2', 'gdT_2', 'NK_3', 'CD8 TEM_3', 'CD4 Proliferating', 'Eryth', 'gdT_4', 'Plasmablast',
#                     'cDC1', 'ASDC_mDC', 'ILC']#original_data
#
# # global_cell_types =['CD4 T', 'CD8 T', 'Mono', 'NK']#l1
#
# select_data = ["Phase",'nCount_ADT',"nFeature_ADT","nFeature_RNA","celltype.l3"]

# ###
# global_cell_types =['Acinar cell', 'Delta cell', 'Endothelial cell', 'TUBA1A+ ductal cell', 'Alpha cell', 'Mast cell', 'Fibroblast', 'Beta cell', 'Pericyte', 'Macrophage', 'Epsilon cell']
# select_data=['percent.mt',
#        'RNA_snn_res.0.8', 'seurat_clusters', 'celltype',
#        'sampling.weight', 'n_genes_by_counts',
#        'total_counts',  'pct_counts_in_top_20_genes',
#        'total_counts_mt',  'pct_counts_mt',
#        'total_counts_ribo',  'pct_counts_ribo',
#        'total_counts_hb', 'pct_counts_hb', 'nUMIs',
#        'mito_perc', 'detected_genes']

##covid
# global_cell_types =['CD4.Tfh', 'CD16_mono', 'CD8.TE', 'CD83_CD14_mono', 'CD14_mono',
#                     'NK_16hi', 'CD4.CM', 'B_naive', 'B_immature', 'MAIT', 'CD8.EM', 'NKT', 'CD8.Naive',
#                     'CD4.Naive', 'gdT', 'B_non-switched_memory', 'B_exhausted', 'DC2', 'HSC_CD38pos', 'CD4.Prolif',
#                     'C1_CD16_mono', 'CD4.IL22', 'CD4.EM', 'Platelets', 'Plasmablast', 'HSC_CD38neg', 'Plasma_cell_IgA', 'pDC',
#                     'NK_56hi', 'DC3', 'CD4.Th1', 'B_switched_memory', 'Plasma_cell_IgG', 'B_malignant', 'NK_prolif', 'Mono_prolif',
#                     'Plasma_cell_IgM', 'HSC_erythroid', 'Treg', 'RBC', 'CD8.Prolif', 'ILC1_3', 'DC1', 'ASDC', 'ILC2']

# select_data =[ 'Centre',
#        'high_confidence', 'length', 'chain', 'v_gene', 'd_gene', 'j_gene',
#        'c_gene', 'full_length', 'productive',  'reads',
#        'umis', 'raw_clonotype_id', 'raw_consensus_id',
#        'Is.Tcell', 'sample_id', 'cells', 'Site', 'doublet',
#        'patient_id', 'batch', 'Resample', 'Sex',
#        'Swab_result', 'Status', 'Smoker', 'Status_on_day_collection',
#        'Status_on_day_collection_summary', 'Days_from_onset', 'time_after_LPS',
#        'Worst_Clinical_Status', 'Outcome',  'Age']
# select_data =[ 'total_counts_mt', 'pct_counts_mt', 'full_clustering',  'Resample',
#  'Collection_Day', 'Sex', 'Age_interval', 'Swab_result', 'Status', 'Smoker', 'Status_on_day_collection',
#  'Status_on_day_collection_summary', 'Days_from_onset', 'Site', 'time_after_LPS', 'Worst_Clinical_Status',
#  'Outcome']

# ###heart
# global_cell_types =['Cardiomyocyte_I', 'Cardiomyocyte_II', 'Cardiomyocyte_III']
# select_data =[ 'disease', 'sex', 'age', 'lvef','cellbender_ncount',
#        'cellbender_ngenes', 'cellranger_percent_mito', 'exon_prop',"cell_type_leiden0.6",
#        'cellbender_entropy', 'cellranger_doublet_scores', 'n_counts',
#        'n_counts_normalized', 'n_counts_normalized_log']
###macro
# select_data =['species','disease', 'organ', 'sex',
#        "cell_type__ontology_label", 'sample','chemistry','paper', 'health',  'indication', 'tissue']
# global_cell_types =['macrophage', 'classical monocyte', 'monocyte', 'conventional dendritic cell', 'Kupffer cell', 'non-classical monocyte', 'alveolar macrophage', 'mature conventional dendritic cell', 'apoptosis fated cell', 'mitotic cell cycle']
# global_cell_types =['AM', 'SAMac', 'Mo_DC_Transitional', 'Kupffer', 'Mo_Classical', 'Mac_CCL2_CCL7',
#                     'Mac_HMOX1', 'Mo_Nonclassical', 'Mac_SLC16A10', 'Resident_Mac_IFN_responsive', 'Mac_RETN',
#                     'Mo_FCN1_CLEC4E', 'Mo_Inflammatory', 'Mac_TIMP1', 'LYZ_KRLG1_high', 'preSAMac_APOE', 'cDC2',
#                     'cDC1', 'Mac_MS4A7', 'Mac_FN1', 'IFN_responsive',
#                     'DC_MHC_high', 'Proliferating', 'Ex_vivo_activated', 'Mature_DCs', 'PPBP_high']
# # cancer
# select_data =['disease','organ',  'library_preparation_protocol', 'sex', 'Condition',"CellType"]
# global_cell_types =['Cancer/Epithelial', 'Plasmablasts', 'T-cells', 'Endothelial', 'PVL cells', 'CAFs', 'Cancer', 'B-cells', 'Myoepithelial', 'Monocytes/Macrophages', 'T-cells Cycling', 'Doublets', 'Monocytes/Macrophages Cycling', 'Mouse 3T3 spike-in', 'Endothelial 1', 'Unassigned', 'Cancer/Epithelial Cycling', 'Endothelial 2', 'NK cells', 'MAST cells', 'pDCs', 'cDCs']

#immue
select_data =['species', 'disease__ontology_label',  'organ__ontology_label',
       'library_preparation_protocol__ontology_label', 'sex', 'Age_range',
       'Manually_curated_celltype']

global_cell_types =['Trm_Th1/Th17',
 'Trm_gut_CD8','Trm_Tgd','Erythrophagocytic macrophages',
 'T_CD4/CD8','Trm/em_CD8','Cycling T&NK','Memory B cells','Tregs','Naive B cells','Plasma cells','Teffector/EM_CD4','Tem/emra_CD8','DC2',
 'Tfh','MNP/B doublets','DC1','ILC3','Tnaive/CM_CD4','Tgd_CRTAM+','Intermediate macrophages','Alveolar macrophages','NK_CD16+','Classical monocytes','Mast cells',
 'Nonclassical monocytes','Tnaive/CM_CD8','MAIT','pDC', 'Cycling','migDC', 'MNP/T doublets','ABCs','NK_CD56bright_CD16-','Intestinal macrophages','T/B doublets','GC_B (I)','GC_B (II)',
 'Tnaive/CM_CD4_activated','Plasmablasts','Progenitor','Erythroid','Pro-B','Pre-B','Megakaryocytes']

global_label_mapping = {celltype: i for i, celltype in enumerate(global_cell_types)}
# global_label_mapping = {celltype: string.ascii_uppercase[i] for i, celltype in enumerate(global_cell_types)}
def get_unique_genes_for_cell(adata, cell_index, top_n=30):
    """Retrieves the top N expressed genes for a given cell."""
    cell_expr = adata[cell_index].X
    cell_expr = cell_expr.toarray().flatten() if hasattr(cell_expr,
                                                         'toarray') else cell_expr.flatten()  # Handle sparse/dense
    top_indices = np.argsort(cell_expr)[-top_n:]
    return {adata.var_names[i]: cell_expr[i] for i in top_indices}

def tokenize(tokenizer,max_length, prompt, answer):
    example = prompt + answer
    prompt = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
    example = torch.tensor(tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
    padding = max_length - example.shape[0]
    # print(padding)
    if padding > 0:
        example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        example = example[:max_length]
    labels = copy.deepcopy(example)
    labels[:len(prompt)] = -1
    example_mask = example.ge(0)
    label_mask = labels.ge(0)
    example[~example_mask] = 0
    labels[~label_mask] = 0
    example_mask = example_mask.float()
    label_mask = label_mask.float()
    return example, labels, example_mask, label_mask

class hPancreasDataSet(Data.Dataset):
    def __init__(self, args,model_path, top_genes=512, use_unique_genes=False, max_length=512,is_eval=False,top_gene_list=None):
        vocab=GeneVocab.from_file(args.vocab_file)
        adata=sc.read(args.data_path)
        self.top_genes = top_genes
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数

            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata.var['highly_variable_rank']).values  # [::-1]  # 获取排序后的基因索引
            sorted_genes = adata.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            with open("select_genes.json", 'w') as f:  # args.output_genes_file是保存基因列表的json文件路径
                json.dump(top_genes, f)
        else:
            adata = adata[:,top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]

        count_matrix = adata.X

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')


        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt=is_eval
    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        # if self.use_unique_genes:
        #     top_genes = get_unique_genes_for_cell(self.adata, cell_index, top_n=self.top_genes)
        # else:
        #     top_genes = get_top_genes_for_cell(self.adata, cell_index, top_n=self.top_genes)
        cell_types_str = "; ".join(self.cell_types)
        prompt = f"A series of genes with their expression levels in a cell. What cell type do you think it is? Here are the possible cell types: {cell_types_str}."

        cell_type_key=self.adata.obs['Celltype'].iloc[idx]
        label = self.label_mapping[cell_type_key]
        answer = f"This gene is {label}: {cell_type_key}."
        example, labels, example_mask, label_mask = tokenize(self.tokenizer,self.max_length,prompt,answer)
        values = self.count_matrix[idx]
        genes = self.gene_ids
        # 获取前 top_genes 个基因的索引
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))

        genes = genes[top_gene_indices]
        values = values[top_gene_indices]

        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        if self.return_prompt:
            return prompt,genes, values, gene_mask,label
        return example, labels, example_mask, genes, values, gene_mask

class MyeloidDataSet(Data.Dataset):
    def __init__(self, args,model_path, top_genes=1024, use_unique_genes=False, max_length=256,is_eval=False,top_gene_list=None):
        vocab=GeneVocab.from_file(args.vocab_file)

        adata=sc.read(args.data_path)
        self.top_genes = top_genes
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数

            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata.var['highly_variable_rank']).values  # [::-1]  # 获取排序后的基因索引
            sorted_genes = adata.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            os.makedirs(args.output_dir, exist_ok=True)
            output_genes_file = os.path.join(args.output_dir, "select_genes.json")  # 拼接文件路径
            with open(output_genes_file, 'w') as f:
                json.dump(top_genes, f)
        else:
            adata = adata[:,top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]

        count_matrix = adata.X

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt=is_eval
    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        # if self.use_unique_genes:
        #     top_genes = get_unique_genes_for_cell(self.adata, cell_index, top_n=self.top_genes)
        # else:
        #     top_genes = get_top_genes_for_cell(self.adata, cell_index, top_n=self.top_genes)
        cell_types_str = "; ".join(self.cell_types)
        prompt = f"A series of genes with their expression levels in a cell. What cell type do you think it is? Here are the possible cell types: {cell_types_str}."

        cell_type_key=self.adata.obs['cell_type'].iloc[idx]
        batch=None
        if "batch" in self.adata.obs.columns:
            batch=self.adata.obs['batch'].iloc[idx]
        label = self.label_mapping[cell_type_key]
        answer = f"This gene is {label}: {cell_type_key}."
        example, labels, example_mask, label_mask = tokenize(self.tokenizer,self.max_length,prompt,answer)
        values = self.count_matrix[idx]
        genes = self.gene_ids

        # 获取前 top_genes 个基因的索引
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))

        genes = genes[top_gene_indices]
        values = values[top_gene_indices]

        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        if self.return_prompt:
            return prompt,genes, values, gene_mask,label,batch
        return example, labels, example_mask, genes, values, gene_mask,batch



class msDataSet(Data.Dataset):
    def __init__(self, args,model_path, top_genes=512, use_unique_genes=False, max_length=256,is_eval=False,top_gene_list=None,transfer=False):
        vocab=GeneVocab.from_file(args.vocab_file)
        adata=sc.read(args.data_path)
        if transfer:
            gene_replace_list=pd.read_csv("/home/xh/data/ms/gene_info.csv")
            id_to_name = dict(zip(gene_replace_list['feature_id'], gene_replace_list['feature_name']))
            # 使用 map 函数替换基因名称
            new_var_names = adata.var_names.to_series().map(id_to_name)
                             # .fillna(adata.var_names.tolist()))
            valid_var_names = new_var_names.dropna()
            valid_indices = valid_var_names.index # 根据有效的索引重新过滤 adata 对象 adata = adata[:, valid_indices]
            adata = adata[:, valid_indices]
        # if not var_names.is_unique: raise ValueError("var_names contains duplicates. Please ensure unique var_names.")
            adata.var_names = valid_var_names
        self.top_genes = top_genes
        top_gene_set = set(top_gene_list)  # 转换为集合以便高效查找
        adata_genes_set = set(adata.var_names.values.astype(str))  # 转换为集合,确保类型为字符串

        common_genes = top_gene_set.intersection(adata_genes_set)
        num_common_genes = len(common_genes)

        print(f"adata 中包含 {num_common_genes} 个 top_gene_list 中的基因")
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数

            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata.var['highly_variable_rank']).values  # [::-1]  # 获取排序后的基因索引
            sorted_genes = adata.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            output_genes_file = os.path.join(args.output_dir, "select_genes.json")  # 拼接文件路径
            with open(output_genes_file, 'w') as f:
                json.dump(top_genes, f)

        else:
            adata = adata[:,top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]
        count_matrix = adata.X.toarray()

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt=is_eval
        self.label_length=len(self.cell_types)
    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        # if self.use_unique_genes:
        #     top_genes = get_unique_genes_for_cell(self.adata, cell_index, top_n=self.top_genes)
        # else:
        #     top_genes = get_top_genes_for_cell(self.adata, cell_index, top_n=self.top_genes)
        cell_types_str = "; ".join(self.cell_types)
        prompt = f"A series of genes with their expression levels in a cell. What cell type do you think it is? Here are the possible cell types: {cell_types_str}."

        cell_type_key=self.adata.obs['celltype'].iloc[idx]
        batch=self.adata.obs['str_batch'].iloc[idx]
        label = self.label_mapping[cell_type_key]
        answer = f"This gene is {label}: {cell_type_key}."
        example, labels, example_mask, label_mask = tokenize(self.tokenizer,self.max_length,prompt,answer)
        values = self.count_matrix[idx]
        genes = self.gene_ids
        # 获取前 top_genes 个基因的索引
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))
        # print(values, genes)
        genes = genes[top_gene_indices]
        values = values[top_gene_indices]

        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        if self.return_prompt:
            return prompt,genes, values, gene_mask,label,

        return example, torch.tensor(label, dtype=torch.long), example_mask, genes, values, gene_mask


class msDataSet2(Data.Dataset):
    def __init__(self, args,model_path, top_genes=512, use_unique_genes=False, max_length=512,is_eval=False,top_gene_list=None,transfer=False):
        vocab=GeneVocab.from_file(args.vocab_file)
        adata=sc.read(args.data_path)
        adata.obs = adata.obs[select_data]
        new_col_names = {}
        for col in select_data:
            if "[" in col and "]" in col:
                new_name = col[col.find("[") + 1:col.find("]")]
                new_col_names[col] = new_name
            else:
                new_col_names[col] = col
        adata.obs = adata.obs.rename(columns=new_col_names)
        self.obs_features = [col for col in adata.obs.columns if col != 'celltype']
        if transfer:
            gene_replace_list=pd.read_csv("/home/xh/data/ms/gene_info.csv")
            id_to_name = dict(zip(gene_replace_list['feature_id'], gene_replace_list['feature_name']))
            new_var_names = adata.var_names.to_series().map(id_to_name)
            valid_var_names = new_var_names.dropna()
            valid_indices = valid_var_names.index # 根据有效的索引重新过滤 adata 对象 adata = adata[:, valid_indices]
            adata = adata[:, valid_indices]
            adata.var_names = valid_var_names
        self.top_genes = top_genes
        top_gene_set = set(top_gene_list)  # 转换为集合以便高效查找
        adata_genes_set = set(adata.var_names.values.astype(str))  # 转换为集合,确保类型为字符串

        common_genes = top_gene_set.intersection(adata_genes_set)
        num_common_genes = len(common_genes)
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数

            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata.var['highly_variable_rank']).values  # [::-1]  # 获取排序后的基因索引
            sorted_genes = adata.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            output_genes_file = os.path.join(args.output_dir, "select_genes.json")  # 拼接文件路径
            with open(output_genes_file, 'w') as f:
                json.dump(top_genes, f)

        else:
            adata = adata[:,top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]
        count_matrix = adata.X.toarray()

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt=is_eval
        self.label_length=len(self.cell_types)
    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        top_genes_select = get_unique_genes_for_cell(self.adata, cell_index)
        genes_str = ", ".join(list(top_genes_select.keys()))  # 只获取基因名
        # prompt = f"This is a general description of a cell in the cerebral cortex. with top genes are {genes_str}. "
        prompt = f"This is a general description of a cell in the cerebral cortex."
        # if self.obs_features:  # 使用 self.obs_features
        #     for feature in self.obs_features:
        #         if feature in self.adata.obs:
        #             feature_value = str(self.adata.obs[feature].iloc[idx])  # 使用 idx 获取值
        #             prompt += f"{feature} of this cell is {feature_value}. "
        cell_types_str = "; ".join(self.cell_types)
        prompt += f"What cell type do you think it is in our brain? Here are the all possible cell types: {cell_types_str}."
        # prompt += f"What cell type do you think it is in our brain? "
        cell_type_key=self.adata.obs['celltype'].iloc[idx]
        label = self.label_mapping[cell_type_key]
        answer = ""
        example, labels, example_mask, label_mask = tokenize(self.tokenizer,self.max_length,prompt,answer)
        values = self.count_matrix[idx]
        genes = self.gene_ids
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))
        genes = genes[top_gene_indices]
        values = values[top_gene_indices]

        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        if self.return_prompt:
            return prompt,genes, values, gene_mask,label,

        return example, torch.tensor(label, dtype=torch.long), example_mask, genes, values, gene_mask


class simulationDataset(Data.Dataset):
    def __init__(self, args,model_path, top_genes=512, use_unique_genes=False, max_length=512,is_eval=False,top_gene_list=None):
        vocab=GeneVocab.from_file(args.vocab_file)
        adata=sc.read(args.data_path)
        train_bool = [x in ['P1', 'P3', 'P4', 'P7'] for x in adata.obs['donor']]
        if is_eval:
            adata = adata[np.invert(train_bool)]  # 直接过滤adata，保留不在训练集中的数据
        else:
            adata = adata[train_bool]
        adata.obs = adata.obs[select_data]
        self.obs_features = [col for col in adata.obs.columns if col != 'celltype.l3']
        self.top_genes = top_genes
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数
            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata.var['highly_variable_rank']).values  # [::-1]  # 获取排序后的基因索引
            sorted_genes = adata.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            output_genes_file = os.path.join(args.output_dir, "select_genes.json")  # 拼接文件路径
            with open(output_genes_file, 'w') as f:
                json.dump(top_genes, f)
        else:
            adata = adata[:,top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]
        count_matrix = adata.X.toarray()

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt=is_eval
        self.label_length=len(self.cell_types)
    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        top_genes_select = get_unique_genes_for_cell(self.adata, cell_index)
        genes_str = ", ".join(list(top_genes_select.keys()))  # 只获取基因名
        prompt = f"This is a general description of a immune cell. with top genes are {genes_str}. "
        if self.obs_features:  # 使用 self.obs_features
            for feature in self.obs_features:
                if feature in self.adata.obs:
                    feature_value = str(self.adata.obs[feature].iloc[idx])  # 使用 idx 获取值
                    prompt += f"{feature} of this cell is {feature_value}. "
        cell_types_str = "; ".join(self.cell_types)
        prompt += f"What immune cell type do you think it is? Here are the all possible cell types: {cell_types_str}."
        cell_type_key=self.adata.obs['celltype.l3'].iloc[idx]
        label = self.label_mapping[cell_type_key]
        example, labels, example_mask, label_mask = tokenize(self.tokenizer,self.max_length,prompt,'')
        values = self.count_matrix[idx]
        genes = self.gene_ids
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))
        genes = genes[top_gene_indices]
        values = values[top_gene_indices]
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        # if self.return_prompt:
        #     return prompt,genes, values, gene_mask,label,
        return example, torch.tensor(label, dtype=torch.long), example_mask, genes, values, gene_mask



class hanpdataset(Data.Dataset):
    def __init__(self, args,model_path, top_genes=512, use_unique_genes=False, max_length=512,is_eval=False,top_gene_list=None,select_label='celltype'):
        vocab=GeneVocab.from_file(args.vocab_file)
        adata=sc.read(args.data_path)
        adata.X = adata.X.toarray()
        adata_transformed = copy.deepcopy(adata)
        adata_transformed.X = np.log1p(np.exp(adata_transformed.X))
        print(f"转换后数据 (稠密数组) 最小值: {np.min(adata_transformed.X)}")
        print(f"转换后数据 (稠密数组) 最大值: {np.max(adata_transformed.X)}")
        adata_transformed.X = np.nan_to_num(adata_transformed.X, nan=0.0)
        adata.obs = adata.obs[select_data]
        self.obs_features = [col for col in adata.obs.columns if col != select_label]
        self.top_genes = top_genes
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata_transformed, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数
            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata_transformed.var['highly_variable_rank']).values
            sorted_genes = adata_transformed.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            output_genes_file = os.path.join(args.output_dir, "select_genes.json")  # 拼接文件路径
            with open(output_genes_file, 'w') as f:
                json.dump(top_genes, f)
        else:
            adata = adata[:,top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]
        count_matrix = adata.X

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.select_label=select_label
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt=is_eval
        self.label_length=len(self.cell_types)
    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        top_genes_select = get_unique_genes_for_cell(self.adata, cell_index)
        genes_str = ", ".join(list(top_genes_select.keys()))  # 只获取基因名
        prompt = f"This is a general description of a cell. with top genes are {genes_str}. "
        if self.obs_features:  # 使用 self.obs_features
            for feature in self.obs_features:
                if feature in self.adata.obs:
                    feature_value = str(self.adata.obs[feature].iloc[idx])  # 使用 idx 获取值
                    prompt += f"{feature} of this cell is {feature_value}. "
        cell_types_str = "; ".join(self.cell_types)
        prompt += f"What cell type do you think it is? Here are the all possible cell types: {cell_types_str}."
        # print(prompt)
        cell_type_key=self.adata.obs[self.select_label].iloc[idx]
        label = self.label_mapping[cell_type_key]
        example, labels, example_mask, label_mask = tokenize(self.tokenizer,self.max_length,prompt,'')
        values = self.count_matrix[idx]
        genes = self.gene_ids
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))
        genes = genes[top_gene_indices]
        values = values[top_gene_indices]
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        return example, torch.tensor(label, dtype=torch.long), example_mask, genes, values, gene_mask
class heartdataset(Data.Dataset):
    def __init__(self, args, model_path, top_genes=512, use_unique_genes=False, max_length=512, is_eval=False,
                 top_gene_list=None,select_label='celltype'):
        vocab = GeneVocab.from_file(args.vocab_file)
        adata = sc.read(args.data_path)
        self.obs_features = [col for col in adata.obs.columns if col != select_label]
        self.top_genes = top_genes
        self.select_label = select_label
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数
            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata.var['highly_variable_rank']).values  # [::-1]  # 获取排序后的基因索引
            sorted_genes = adata.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            output_genes_file = os.path.join(args.output_dir, "select_genes.json")  # 拼接文件路径
            with open(output_genes_file, 'w') as f:
                json.dump(top_genes, f)
        else:
            adata = adata[:, top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]
        count_matrix = adata.X.toarray()

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt = is_eval
        self.label_length = len(self.cell_types)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        top_genes_select = get_unique_genes_for_cell(self.adata, cell_index)
        genes_str = ", ".join(list(top_genes_select.keys()))  # 只获取基因名
        prompt = f"This is a general description of a heart cell. with top genes are {genes_str}. "
        if self.obs_features:  # 使用 self.obs_features
            for feature in self.obs_features:
                if feature in self.adata.obs:
                    feature_value = str(self.adata.obs[feature].iloc[idx])  # 使用 idx 获取值
                    prompt += f"{feature} of this cell is {feature_value}. "
        cell_types_str = "; ".join(self.cell_types)
        prompt += f"You need to distinguish cardiomyocytes in non-failing hearts from those in hypertrophic or dilated cardiomyopathy samples. Here are the all possible cell types: {cell_types_str}."
        cell_type_key = self.adata.obs[self.select_label].iloc[idx]
        label = self.label_mapping[cell_type_key]
        example, labels, example_mask, label_mask = tokenize(self.tokenizer, self.max_length, prompt, '')
        values = self.count_matrix[idx]
        genes = self.gene_ids
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))
        genes = genes[top_gene_indices]
        values = values[top_gene_indices]
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        return example, torch.tensor(label, dtype=torch.long), example_mask, genes, values, gene_mask

class Macrophagesdataset(Data.Dataset):
    def __init__(self, args, model_path, top_genes=1024, use_unique_genes=False, max_length=512, is_eval=False,
                 top_gene_list=None,select_label='celltype'):
        vocab = GeneVocab.from_file(args.vocab_file)
        adata = sc.read(args.data_path)
        self.obs_features = [col for col in select_data if col != select_label]
        print(self.obs_features)
        self.top_genes = top_genes
        self.select_label = select_label
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数
            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata.var['highly_variable_rank']).values  # [::-1]  # 获取排序后的基因索引
            sorted_genes = adata.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            output_genes_file = os.path.join(args.output_dir, "select_genes.json")  # 拼接文件路径
            with open(output_genes_file, 'w') as f:
                json.dump(top_genes, f)
        else:
            adata = adata[:, top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]
        count_matrix = adata.X.toarray()

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt = is_eval
        self.label_length = len(self.cell_types)

    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        top_genes_select = get_unique_genes_for_cell(self.adata, cell_index)
        genes_str = ", ".join(list(top_genes_select.keys()))  # 只获取基因名
        prompt = f"This is a general description of a human macrophages cell. with top genes are {genes_str}. "
        if self.obs_features:  # 使用 self.obs_features
            for feature in self.obs_features:
                if feature in self.adata.obs:
                    feature_value = str(self.adata.obs[feature].iloc[idx])  # 使用 idx 获取值
                    prompt += f"{feature} of this cell is {feature_value}. "
        cell_types_str = "; ".join(self.cell_types)
        prompt += f"What cell type of it? Here are the all possible cell types: {cell_types_str}."
        cell_type_key = self.adata.obs[self.select_label].iloc[idx]
        label = self.label_mapping[cell_type_key]
        example, labels, example_mask, label_mask = tokenize(self.tokenizer, self.max_length, prompt, '')
        values = self.count_matrix[idx]
        genes = self.gene_ids
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))
        genes = genes[top_gene_indices]
        values = values[top_gene_indices]
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        return example, torch.tensor(label, dtype=torch.long), example_mask, genes, values, gene_mask

class cancerdataset(Data.Dataset):
    def __init__(self, args, model_path, top_genes=1024, use_unique_genes=False, max_length=512, is_eval=False,
                 top_gene_list=None,select_label='celltype'):
        vocab = GeneVocab.from_file(args.vocab_file)
        adata = sc.read(args.data_path)
        self.obs_features = [col for col in select_data if col != select_label]
        print(self.obs_features)
        self.top_genes = top_genes
        self.select_label = select_label
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数
            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata.var['highly_variable_rank']).values  # [::-1]  # 获取排序后的基因索引
            sorted_genes = adata.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            output_genes_file = os.path.join(args.output_dir, "select_genes.json")  # 拼接文件路径
            with open(output_genes_file, 'w') as f:
                json.dump(top_genes, f)
        else:
            adata = adata[:, top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]
        count_matrix = adata.X.toarray()

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt = is_eval
        self.label_length = len(self.cell_types)

    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        top_genes_select = get_unique_genes_for_cell(self.adata, cell_index)
        genes_str = ", ".join(list(top_genes_select.keys()))  # 只获取基因名
        prompt = f"This is a general description of a human cell. with top genes are {genes_str}. "
        if self.obs_features:  # 使用 self.obs_features
            for feature in self.obs_features:
                if feature in self.adata.obs:
                    feature_value = str(self.adata.obs[feature].iloc[idx])  # 使用 idx 获取值
                    prompt += f"{feature} of this cell is {feature_value}. "
        cell_types_str = "; ".join(self.cell_types)
        prompt += f"What cell type of it? Here are the all possible cell types: {cell_types_str}."
        cell_type_key = self.adata.obs[self.select_label].iloc[idx]
        label = self.label_mapping[cell_type_key]
        example, labels, example_mask, label_mask = tokenize(self.tokenizer, self.max_length, prompt, '')
        values = self.count_matrix[idx]
        genes = self.gene_ids
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))
        genes = genes[top_gene_indices]
        values = values[top_gene_indices]
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        return example, torch.tensor(label, dtype=torch.long), example_mask, genes, values, gene_mask


class immunedataset(Data.Dataset):
    def __init__(self, args, model_path, top_genes=1024, use_unique_genes=False, max_length=512, is_eval=False,
                 top_gene_list=None,select_label='celltype'):
        vocab = GeneVocab.from_file(args.vocab_file)
        adata = sc.read(args.data_path)
        self.obs_features = [col for col in select_data if col != select_label]
        print(self.obs_features)
        self.top_genes = top_genes
        self.select_label = select_label
        if top_gene_list is None:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=len(adata.var_names))  # 计算所有基因的变异系数
            # 根据变异系数排序基因 (从大到小)
            sorted_indices = np.argsort(adata.var['highly_variable_rank']).values  # [::-1]  # 获取排序后的基因索引
            sorted_genes = adata.var_names[sorted_indices].values
            valid_genes = [gene for gene in sorted_genes if gene in vocab]
            adata = adata[:, valid_genes]
            top_genes = valid_genes[:top_genes]  # 获取top_genes个基因
            adata = adata[:, top_genes]  # 缩减 adata
            output_genes_file = os.path.join(args.output_dir, "select_genes.json")  # 拼接文件路径
            with open(output_genes_file, 'w') as f:
                json.dump(top_genes, f)
        else:
            adata = adata[:, top_gene_list]
        self.adata = adata  # 这行最好放在排序和过滤之后
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var_names]
        count_matrix = adata.X.toarray()

        self.count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        self.gene_ids = np.array(adata.var["id_in_vocab"])
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.use_unique_genes = use_unique_genes
        self.max_length = max_length
        self.cell_types = global_cell_types  # 使用全局的细胞类型
        self.label_mapping = global_label_mapping
        self.return_prompt = is_eval
        self.label_length = len(self.cell_types)

    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_index = self.adata.obs_names[idx]
        top_genes_select = get_unique_genes_for_cell(self.adata, cell_index)
        genes_str = ", ".join(list(top_genes_select.keys()))  # 只获取基因名
        # prompt = f"This is a general description of a immune cell. with top genes are {genes_str}. "
        prompt ="This is a general description of a immune cell."
        if self.obs_features:  # 使用 self.obs_features
            for feature in self.obs_features:
                if feature in self.adata.obs:
                    feature_value = str(self.adata.obs[feature].iloc[idx])  # 使用 idx 获取值
                    prompt += f"{feature} of this cell is {feature_value}. "
        cell_types_str = "; ".join(self.cell_types)
        prompt += f"What cell type of it? Here are the all possible cell types: {cell_types_str}."
        # print(prompt)
        # prompt += f"What cell type of it?"
        cell_type_key = self.adata.obs[self.select_label].iloc[idx]
        label = self.label_mapping[cell_type_key]
        example, labels, example_mask, label_mask = tokenize(self.tokenizer, self.max_length, prompt, '')
        values = self.count_matrix[idx]
        genes = self.gene_ids
        top_gene_indices = np.arange(min(self.top_genes, len(genes)))
        genes = genes[top_gene_indices]
        values = values[top_gene_indices]
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).half()
        gene_mask = torch.zeros_like(genes, dtype=torch.bool)
        return example, torch.tensor(label, dtype=torch.long), example_mask, genes, values, gene_mask