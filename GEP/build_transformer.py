import torch

import json
from GEP import ModelArgs, Tokenizer, Transformer
from GEP.adapter import set_Llama_Adapter

from pathlib import Path


def _load_and_redistribute_checkpoint(llama_model_path, model_name):
    with open(Path(llama_model_path) / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name == '7B':
        checkpoint = torch.load(llama_model_path + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    loaded = []
    for x in checkpoints:
        print('loading from', x)
        loaded.append(torch.load(x, map_location='cpu'))

    full_state_dict = {}
    split_dims = {}

    def add_weight_with_split_dim(name, dim):
        if dim < 0:  # bcast without split
            full_state_dict[name] = loaded[0][name].clone()
        else:
            full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
        for x in loaded:
            del x[name]
        split_dims[name] = dim

    add_weight_with_split_dim('tok_embeddings.weight', 1)
    add_weight_with_split_dim('norm.weight', -1)
    add_weight_with_split_dim('output.weight', 0)
    for i in range(params['n_layers']):
        print('gathering layer %d of %d' % (i, params['n_layers']))
        layer_prefix = f'layers.{i}.'
        bcast_names = [
            'attention_norm.weight',
            'ffn_norm.weight',
        ]
        column_parallel_names = [
            'attention.wq.weight',
            'attention.wk.weight',
            'attention.wv.weight',
            'feed_forward.w1.weight',
            'feed_forward.w3.weight',
        ]
        row_parallel_names = [
            'attention.wo.weight',
            'feed_forward.w2.weight',
        ]
        for key in bcast_names:
            add_weight_with_split_dim(layer_prefix + key, -1)
        for key in column_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 0)
        for key in row_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 1)

    checkpoint = full_state_dict

    return checkpoint, tokenizer, params


def create_model(args):
    llama_model_path = args.llama_model_path
    model_name = args.llm_model

    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(llama_model_path, model_name)

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len, max_batch_size=64, hidden_proj=args.hidden_proj,
        emb=args.emb, is_train=args.is_train, **params
    )
    model_args.model_file=args.model_file
    model_args.label_length = args.label_length
    model_args.config_file=args.config_file
    model_args.vocab_file=args.vocab_file
    model_args.use_batch_labels=args.use_batch_labels
    model_args.vocab_size = tokenizer.n_words

    if args.cpu_load:
        # cpu load is slow, but is freindly for GPU with limited memory.
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    llama = Transformer(model_args)



    torch.set_default_tensor_type(torch.FloatTensor)

    llama.load_state_dict(checkpoint, strict=False)


    set_Llama_Adapter(llama, s=args.adapter_scale, gradient_checkpointing=args.gradient_checkpointing)

    learnable_keys = ['adapter']
    total = 0.
    trainable_names = []
    for name, param in llama.named_parameters():
        for key in learnable_keys:
            if key in name:
                param.requires_grad = True
                param.data = param.data.float()
                total += param.nelement()
                trainable_names.append(name)
            else:
                param.requires_grad = False
    llama.classifier.requires_grad=True
    llama.backbone.requires_grad = True
    # print(trainable_names)
    print('  + Number of trainable params: %.2fM' % (total / 1e6))
    return llama
