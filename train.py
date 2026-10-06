"""Train U-Net with BCE, Dice, or their sum; ignore ambiguous trimap pixels."""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from data import loaders
from model import UNet


def segmentation_loss(logits, target, valid, kind):
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * valid).sum() / valid.sum()
    probabilities = logits.sigmoid() * valid
    target = target * valid
    dice = 1 - (2 * (probabilities * target).sum() + 1) / (probabilities.sum() + target.sum() + 1)
    return {'bce': bce, 'dice': dice, 'combined': bce + dice}[kind]


def run_epoch(model, loader, device, loss_kind, optimizer=None):
    model.train(optimizer is not None)
    total_loss = total_images = intersection = union = predicted = target_total = 0
    for images, targets, valid in loader:
        images, targets, valid = images.to(device), targets.to(device), valid.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = segmentation_loss(logits, targets, valid, loss_kind)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        predictions = (logits.sigmoid() >= .5) & valid.bool()
        truth = targets.bool() & valid.bool()
        intersection += (predictions & truth).sum().item()
        union += (predictions | truth).sum().item()
        predicted += predictions.sum().item()
        target_total += truth.sum().item()
        total_loss += loss.item() * len(images)
        total_images += len(images)
    return {'loss': total_loss / total_images, 'iou': intersection / max(union, 1),
            'dice': 2 * intersection / max(predicted + target_total, 1)}


def save_examples(model, loader, device, output):
    images, targets, valid = next(iter(loader))
    with torch.no_grad():
        predictions = (model(images.to(device)).sigmoid().cpu() >= .5).float()
    count = min(3, len(images))
    fig, axes = plt.subplots(count, 3, figsize=(9, 3 * count), squeeze=False)
    for i in range(count):
        axes[i, 0].imshow(images[i].permute(1, 2, 0))
        axes[i, 1].imshow(targets[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].imshow(predictions[i, 0], cmap='gray', vmin=0, vmax=1)
        for axis, title in zip(axes[i], ('Image', 'Target', 'Prediction')):
            axis.set_title(title)
            axis.axis('off')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', choices=('bce', 'dice', 'combined'), default='combined')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('runs/combined'))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size % 16:
        parser.error('epochs/batch-size must be positive and size must be divisible by 16')
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    train_loader, validation_loader, test_loader = loaders(
        args.data_dir, args.batch_size, args.size, args.seed, workers=args.workers)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, args.loss, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, args.loss)
        row = {'epoch': epoch, **{f'train_{k}': v for k, v in train_metrics.items()},
               **{f'validation_{k}': v for k, v in validation_metrics.items()}}
        history.append(row)
        if validation_metrics['iou'] > best_iou:
            best_iou = validation_metrics['iou']
            best_epoch = epoch
            torch.save(model.state_dict(), args.out / 'best.pt')
        print(row, flush=True)
    with (args.out / 'history.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    model.load_state_dict(torch.load(args.out / 'best.pt', map_location=device, weights_only=True))
    test_metrics = run_epoch(model, test_loader, device, args.loss)
    save_examples(model, test_loader, device, args.out / 'examples.png')
    result = {'loss': args.loss, 'best_epoch': best_epoch, 'validation_iou': best_iou,
              **{f'test_{k}': v for k, v in test_metrics.items()}}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
