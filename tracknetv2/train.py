import torch
import argparse
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from tracknetv2.model import TrackNetV2
from tracknetv2.util import CustomLoss, OPTIMIZERS, find_pos


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, heatmaps):
        super().__init__()

        self.imgs = [torch.from_numpy(x) for x in imgs]
        self.heatmaps = [torch.from_numpy(x) for x in heatmaps]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]. self.heatmaps[idx]


def collate(inputs):
    imgs, heatmaps = list(zip(*inputs))
    imgs = torch.vstack(imgs)
    heatmaps = torch.vstack(heatmaps)
    return imgs, heatmaps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="train", help="Path to directory with train data.")
    parser.add_argument("--val_data", type=str, default="val", help="Path to directory with val data.")
    parser.add_argument("--save_path", type=str, default="weights", help="Path to save model weights.")
    parser.add_argument("--logdir", type=str, default="mylogs", help="Path to save training logs.")
    parser.add_argument("--train_config", type=str, default="configs/train_config.json", help="Path train config.")
    parser.add_argument("--model_config", type=str, default="configs/model_config.json", help="Path model config.")
    return parser.parse_args()


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    train_config,
    save_path,
    writer,
    model_v
):

    grad_accum = train_config['accumulate_grad_batches']
    fp16 = train_config['fp16']

    best_metric = None
    best_sd = None

    total_step = 0

    for epoch in tqdm(range(train_config['epochs'])):
        pbar = tqdm(train_loader, total=len(train_loader), unit='batch', leave=False)

        model.train()
        cur_loss = 0
        losses = []
        for batch_idx, batch in enumerate(pbar):
            imgs, target = batch
            imgs = imgs.to(device)
            target = target.to(device)

            if fp16:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    preds = model(imgs)
                    loss = criterion(target, preds)
                    loss /= grad_accum
            else:
                preds = model(imgs)
                loss = criterion(target, preds)
                loss /= grad_accum

            loss.backward()
            cur_loss += loss.detach().cpu().item()

            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=cur_loss)
                writer.add_scalar("loss", cur_loss, total_step)
                losses.append(cur_loss)
                cur_loss = 0
                total_step += 1

        writer.add_scalar("epoch_loss", np.mean(losses), total_step)

        val_metric, val_loss = evaluate(model, val_loader, criterion, device, train_config)
        writer.add_scalar(train_config['metric'], val_metric, total_step)
        writer.add_scalar("val_loss", val_loss, total_step)

        if best_metric is None or (best_metric > val_metric if train_config['minimize_metric'] else
                                   best_metric < val_metric):
            best_metric = val_metric
            best_sd = model.state_dict()
            save_model(best_sd, epoch, total_step, save_path, best_metric, train_config['metric'], model_v)

def evaluate(
    model,
    val_loader,
    criterion,
    device,
    train_config
):
    fp16 = train_config['fp16']
    model.eval()
    val_losses = []
    targets = []
    preds = []
    for batch in tqdm(val_loader, leave=False):
        imgs, target = batch
        imgs = imgs.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(False):
            if fp16:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    pred = model(imgs)
            else:
                pred = model(imgs)
            loss = criterion(target, pred).cpu().item()
        targets.append(target.cpu().numpy())
        preds.append(pred.cpu().numpy())
        val_losses.append(loss)

    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(preds, axis=0)

    if train_config['metric'] == 'loss':
        metric = np.mean(val_losses)
    else:
        metric = compute_metric(preds, targets, train_config['metric'], model.out, train_config['tol'])

    return metric, np.mean(val_losses)


def compute_metric(preds, targets, metric, out, tol):
    if metric not in ['F1', 'Recall', 'Precision', 'Accuracy']:
        raise NotImplementedError(f"Sorry, but {metric} metric currently is not implemented.")

    TP = TN = FP = FN = 0

    for pp, tt in zip(preds, targets):
        for p, t in (zip(pp, tt) if out > 1 else [pp[-1], tt[0]]):
            p_max, t_max = np.amax(p), np.amax(t)
            if p_max == 0 and np.amax(t) == 0:
                TN += 1
            elif np.amax(p) > 0 and np.amax(t) == 0:
                FP += 1
            elif np.amax(p) == 0 and np.amax(t) > 0:
                FN += 1
            else:
                x_p, y_p = find_pos(p[None, ...])
                x_t, y_t = find_pos(t[None, ...])
                dist = np.sqrt((x_p - x_t)**2 + (y_p - y_t)**2)
                if dist >= tol:
                    TP += 1
                else:
                    FP += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    accuracy = (TP + TN) / (TP + TN + FN + FP) if (TP + TN + FN + FP) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    metrics = {
        'F1': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }
    return metrics[metric]


def save_model(sd, epoch, step, path, metric, metric_name, model_v):
    os.makedirs(os.path.join(path, model_v), exist_ok=True)
    files = os.listdir(os.path.join(path, model_v))
    if len(files) == 1:
        os.remove(files[0])
    torch.save(sd, os.path.join(path, model_v, f"model_{epoch=}_{step=}_{metric_name}={metric}.pt"))


def main(args):
    with open(os.path.join(args.train_data, 'imgs.npy'), 'rb') as f:
        train_imgs = np.load(f)
    with open(os.path.join(args.train_data, 'heatmaps.npy'), 'rb') as f:
        train_heatmaps = np.load(f)
    train_dataset = ImgDataset(train_imgs, train_heatmaps)

    with open(os.path.join(args.val_data, 'imgs.npy'), 'rb') as f:
        val_imgs = np.load(f)
    with open(os.path.join(args.val_data, 'heatmaps.npy'), 'rb') as f:
        val_heatmaps = np.load(f)
    val_dataset = ImgDataset(val_imgs, val_heatmaps)

    with open(args.train_config):
        train_config = json.load(f)

    train_dataloader = torch.utils.data.Dataloader(
        train_dataset, shuffle=True, batch_size=train_config['train_bs'], num_workers=8, pin_memory=True,
        collate_fn=collate
    )

    val_dataloader = torch.utils.data.Dataloader(
        val_dataset, shuffle=False, batch_size=train_config['val_bs'], num_workers=8, pin_memory=True,
        collate_fn=collate
    )

    with open(args.model_config) as f:
        model_config = json.load(f)
    model = TrackNetV2(model_config['height'], model_config['width'], model_config['out'], model_config['dropout'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    if train_config['compile']:
        model = torch.compile(model, mode='default')

    criterion = CustomLoss()
    if train_config['optimizer_config'] is not None:
        if train_config['optimizer_config']['name'] in OPTIMIZERS:
            optimizer = OPTIMIZERS[train_config['optimizer_config']['name']]
            train_config['optimizer_config'].pop('name')
        else:
            raise NotImplementedError(f"Currently there are no support for {train_config['optimizer_config']['name']}\
             optimizer.\nAvailable optimizers : {list(OPTIMIZERS.keys())}.")
    else:
        optimizer = OPTIMIZERS['adadelta']

    optimizer = optimizer(model.parameters(), **train_config['optimizer_config'])

    os.makedirs(args.logdir, exist_ok=True)
    model_v = str(len(os.listdir(args.logdir)))
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, model_v),
                           flush_secs=30, max_queue=100)

    train(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        device,
        train_config,
        args.save_path,
        writer,
        model_v
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
