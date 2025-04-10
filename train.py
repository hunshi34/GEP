import os
import argparse
import datetime
import json
import time
import numpy as np
from pathlib import Path
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch

from util.datasets import  (hPancreasDataSet,MyeloidDataSet,msDataSet,msDataSet2,simulationDataset,hanpdataset,heartdataset,
                            Macrophagesdataset,cancerdataset,immunedataset)
from GEP.build import create_model


# import bitsandbytes as bnb

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='./llama', type=str,
                        help='path of llama model')

    parser.add_argument('--llm_model', default='7B', type=str, metavar='MODEL',
                        help='Name of llm model to train')
    parser.add_argument('--cpu_load',  action='store_true',   help='load the model on cpu and avoid OOM on gpu')
    parser.add_argument('--emb', type=int, default=512)
    parser.add_argument('--adapter_dim', type=int, default=8, metavar='LENGTH', help='the dims of adapter layer')

    parser.add_argument('--hidden_proj', type=int, default=128, metavar='LENGTH',
                        help='the visual adapter dim')

    parser.add_argument('--adapter_scale', type=float, default=1., metavar='LENGTH', help='the scales of adapter layer')

    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH',
                        help='the maximum sequence length')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='clip gradient',
                        help='clips gradient norm of an iterable of parameters')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='saving memory costs via gradient_checkpointing')
    parser.add_argument('--warmup_epochs', type=float, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/instruction_dataset/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--use_batch_labels', default=0, type=int)

    # datasets
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE', 'Q-A', 'QM-A', 'Q-AL', 'QM-EA'
                        ],
                        help='prompt format template')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--caption_file', type=str, default='./data/captions.json')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--do_pretrain', action='store_true', help='pre-train on large scale vl instruction')
    parser.add_argument('--dataset', type=str, default='sqa')
    parser.add_argument('--vocab_file', type=str, default="")
    parser.add_argument('--config_file', type=str, default="")
    parser.add_argument('--model_file', type=str, default="")
    parser.add_argument('--top_genes', type=int, default=256)
    parser.add_argument('--label_length', type=str, default=None)
    parser.add_argument('--select_label', type=str, default="cell_type")
    parser.add_argument('--normal_scale', type=float, default=1)
    return parser


def main(args):
    misc.init_distributed_mode(args)

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print(device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    args.is_train = True
    # with open("/home/xh/select_genes.json", "r") as f:
    #     top_gene_list = json.load(f)
    if "heart" in args.data_path:
        dataset_train = heartdataset(args, args.llama_model_path, args.top_genes, select_label=args.select_label)
    elif "Macrophages" in args.data_path:
        dataset_train=Macrophagesdataset(args, args.llama_model_path, args.top_genes, select_label=args.select_label)
    elif "cancer" in args.data_path:
        dataset_train=cancerdataset(args, args.llama_model_path, args.top_genes,select_label=args.select_label)
    elif "immune" in args.data_path:
        dataset_train=immunedataset(args, args.llama_model_path,args.top_genes,select_label=args.select_label)
    else:
        dataset_train = hanpdataset(args, args.llama_model_path, args.top_genes,select_label=args.select_label)
    args.label_length=dataset_train.label_length
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    # print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = create_model(args)
    model.to(device)
    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name}, dtype: {param.dtype}")
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # print("actual lr: %.2e" % args.lr)
    #
    # print("accumulate grad iterations: %d" % args.accum_iter)
    # print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        # print(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)

    # following qlora: apply paged optimizer
    # optimizer = bnb.optim.AdamW32bit(param_groups, lr=args.lr, betas=(0.9, 0.95),is_paged=True) #
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)

    # mixed precision scaler
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            backbone_path = os.path.join(args.output_dir, f"backbone_epoch_{epoch}.pt")  # backbone 保存路径
            torch.save(model.module.backbone.state_dict(), backbone_path)
            classifier_path = os.path.join(args.output_dir, f"classifier_epoch_{epoch}.pt")
            torch.save(model.module.classifier.state_dict(), classifier_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
