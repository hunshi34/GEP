import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    all_preds = []
    all_labels = []
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    prefix_img = torch.tensor(data_loader.dataset.tokenizer.encode("Image: ", bos=False, eos=False), dtype=torch.int64)
    prefix_nonimg = torch.tensor(data_loader.dataset.tokenizer.encode("Image: N/A", bos=False, eos=False),
                                 dtype=torch.int64)

    for data_iter_step, data_tuple in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        if len(data_tuple) == 7:  # 包含 batch 参数
            examples, labels, example_mask, src, value, src_key_padding_mask, batch = data_tuple

        elif len(data_tuple) == 6:  # 不包含 batch 参数
            examples, labels, example_mask, src, value, src_key_padding_mask = data_tuple
            batch=None
        else:
            raise ValueError(f"Unexpected number of elements from dataloader: {len(data_tuple)}")
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        prefix_img = prefix_img.to(examples.device)
        prefix_nonimg = prefix_nonimg.to(examples.device)
        c_loss,pred,labels = model(examples, labels,src=src,value=value,src_key_padding_mask=src_key_padding_mask,batch=batch, prefix_img=prefix_img, prefix_nonimg=prefix_nonimg,)
        all_preds.extend(pred.cpu().tolist())  # Move to CPU to avoid CUDA memory issues
        all_labels.extend(labels.cpu().tolist())
        if torch.isnan(c_loss):
            print('nan')
            c_loss = torch.nan_to_num(c_loss) * 0
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=args.clip_grad)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"Epoch Accuracy: {accuracy:.4f}")
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module,
                  data_loader: Iterable, optimizer: torch.optim.Optimizer,
                  device: torch.device, epoch: int, loss_scaler,
                  log_writer=None,
                  args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        with torch.no_grad():
            c_loss = model(examples, labels)
        loss = c_loss
        loss_value = loss.item()

        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
