import glob

import torch
import torchvision
import argparse
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from tracknetv2.model import TrackNetV2
from tracknetv2.util import CustomLoss, OPTIMIZERS, find_pos
from tracknetv2.dataset import generate_heat_map


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, width, height, sigma, mag):
        super().__init__()

        self.width = width
        self.height = height
        self.sigma = sigma
        self.mag = mag
        self.imgs = []
        self.heatmaps = []
        for dir_path in tqdm(glob.glob(data_path + '/*'), leave=False):
            tmp = []
            for img in glob.glob(dir_path + '/*.jpg'):
                tmp.append(torchvision.io.read_image(img))
            self.imgs.append(torch.cat(tmp, dim=0))
            with open(os.path.join(dir_path, "heatmap.txt")) as f:
                heatmap = f.read().strip().split('\t')
            heatmap = [tuple(int(y.strip()) for y in x.split(',')) for x in heatmap]
            self.heatmaps.append(heatmap)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        tmp = self.heatmaps[idx]
        heatmaps = np.concatenate([generate_heat_map(self.width, self.height, x[0], x[1],
                                                     self.sigma, self.mag)[None, ...] for x in tmp], axis=0)
        return self.imgs[idx], heatmaps


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
    with open(args.train_config) as f:
        train_config = json.load(f)

    with open(args.model_config) as f:
        model_config = json.load(f)

    print("Reading train data..")
    train_dataset = ImgDataset(args.train_data, model_config['width'], model_config['height'],
                               train_config['sigma'], train_config['mag'])
    print("Finish reading train data.")

    print("Reading val data..")
    val_dataset = ImgDataset(args.val_data, model_config['width'], model_config['height'],
                             train_config['sigma'], train_config['mag'])
    print("Finish reading train data.")

    train_dataloader = torch.utils.data.Dataloader(
        train_dataset, shuffle=True, batch_size=train_config['train_bs'], num_workers=8, pin_memory=True,
        collate_fn=collate
    )

    val_dataloader = torch.utils.data.Dataloader(
        val_dataset, shuffle=False, batch_size=train_config['val_bs'], num_workers=8, pin_memory=True,
        collate_fn=collate
    )

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
