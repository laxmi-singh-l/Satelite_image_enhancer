#!/usr/bin/env python3
"""Train a multi-band satellite super-resolution model.

Two architectures (the statement allows CNN / Generative / Transformer teams):

- ``edsr``  — CNN baseline (EDSR), pixel + optional VGG perceptual loss.
- ``esrgan`` — Generative model: EDSR generator + adversarial ESRGAN-style
  discriminator, adds a geometric (spectral-angle) consistency term so bands
  stay spectrally faithful (SAM/ERGAS-friendly).

Uncertainty-aware training: pass ``--dropout`` (e.g. 0.05) to insert dropout
into the generator, enabling Monte-Carlo dropout uncertainty maps at inference
(``drawing/super_resolve.py --uncertainty``).

Run from the project root:

    python -m training.train --images-dir data_image/sr_hr --scale 4 \\
        --model esrgan --dropout 0.05 --spectral 0.1 --epochs 60 --batch-size 8

Checkpoints are written to ``checkpoints/satelite_sr.pt`` (latest) and
``checkpoints/satelite_sr_best.pt`` (best PSNR on validation).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.metrics import psnr, ssim
from models.enhance import SuperResolutionEDSR
from models.gan import ESRGANDiscriminator
from training.dataset import build_sr_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
LOG = logging.getLogger('train')


class PerceptualLoss(nn.Module):
    """VGG19 feature L1 loss (lightweight LPIPS-style). Only used for RGB."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = None
        self._layers = [1, 6, 11, 18, 25]
        try:
            from torchvision.models import vgg19, VGG19_Weights
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
            for p in model.parameters():
                p.requires_grad_(False)
            self.model = model
        except Exception as exc:
            LOG.warning('Perceptual loss unavailable (%s); using pixel loss only', exc)

    def forward(self, x, y):
        if self.model is None:
            return torch.tensor(0.0, device=x.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        xa, ya = (x - mean) / std, (y - mean) / std
        loss = torch.tensor(0.0, device=x.device)
        feats_x, feats_y = [], []
        for i, layer in enumerate(self.model):
            xa = layer(xa)
            ya = layer(ya)
            if i in self._layers:
                feats_x.append(xa)
                feats_y.append(ya)
        for a, b in zip(feats_x, feats_y):
            loss = loss + F.l1_loss(a, b)
        return loss / len(feats_x)


def spectral_angle_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean spectral angle (SAM) as a differentiable loss over band vectors."""
    if a.shape[1] < 2:
        return torch.zeros((), device=a.device)
    av = a.flatten(2)  # (B, C, H*W)
    bv = b.flatten(2)
    num = (av * bv).sum(dim=1)
    den = av.norm(2, dim=1) * bv.norm(2, dim=1) + eps
    cos = (num / den).clamp(-1.0 + eps, 1.0 - eps)
    return (1.0 - cos).mean()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnrs, ssims = [], []
    for batch_idx, (lr, hr) in enumerate(loader):
        lr, hr = lr.to(device), hr.to(device)
        out = model(lr)
        out_np = out.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        hr_np = hr.permute(0, 2, 3, 1).cpu().numpy()
        for o, h in zip(out_np, hr_np):
            psnrs.append(psnr(h, o))
            ssims.append(ssim(h, o))
        if batch_idx >= 31:
            break
    model.train()
    mean_psnr = float(np.mean(psnrs)) if psnrs else float('nan')
    mean_ssim = float(np.mean(ssims)) if ssims else float('nan')
    return mean_psnr, mean_ssim


def save_checkpoint(path, weights, optimizer, scheduler, epoch, hparams,
                    metrics, is_best=False):
    state = {'model': weights, 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict() if scheduler else None,
             'epoch': epoch, 'metrics': metrics, 'hparams': hparams}
    torch.save(state, path)
    tag = 'BEST ' if is_best else ''
    LOG.info('%sCheckpoint saved: %s (epoch %d, PSNR %.2f)',
             tag, path, epoch, metrics.get('psnr', 0))


def parse_args():
    p = argparse.ArgumentParser(description='Satellite super-resolution training')
    p.add_argument('--manifest', type=str, default=None, help='CSV with lr,hr columns')
    p.add_argument('--images-dir', type=str, default=None, help='HR-only images dir (LR synthesised)')
    p.add_argument('--root', type=str, default=None, help='Root for relative manifest paths')
    p.add_argument('--model', choices=['edsr', 'esrgan'], default='edsr',
                   help='edsr = CNN baseline, esrgan = generative GAN')
    p.add_argument('--scale', type=int, default=4, help='Upscale factor (10m -> 2.5m for 4)')
    p.add_argument('--patch', type=int, default=256, help='HR patch size')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--feats', type=int, default=256, help='EDSR feature channels')
    p.add_argument('--blocks', type=int, default=32, help='EDSR residual blocks')
    p.add_argument('--dropout', type=float, default=0.0,
                   help='Generator dropout p (e.g. 0.05) to enable MC-dropout uncertainty')
    p.add_argument('--perceptual', type=float, default=0.1,
                   help='VGG perceptual loss weight (disabled for <3 bands)')
    p.add_argument('--spectral', type=float, default=0.1,
                   help='Spectral-angle (SAM) consistency loss weight (bands>=2)')
    p.add_argument('--gan', type=float, default=0.01,
                   help='Adversarial loss weight (esrgan only)')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--patience', type=int, default=12, help='Early stopping epochs')
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--out', type=str, default='checkpoints',
                   help='Checkpoint directory')
    p.add_argument('--limit-train', type=int, default=None,
                   help='Cap training samples (diagnostics)')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info('Device: %s', device)

    train_ds, val_ds, _, n_channels = build_sr_datasets(
        manifest=args.manifest, images_dir=args.images_dir, root=args.root,
        scale=args.scale, hr_patch_size=args.patch,
    )
    if args.limit_train:
        train_ds.samples = train_ds.samples[:args.limit_train]
    LOG.info('Train: %d  Val: %d  Bands: %d  Scale: %dx  Model: %s',
             len(train_ds), len(val_ds), n_channels, args.scale, args.model)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

    generator = SuperResolutionEDSR(n_colors=n_channels, n_feats=args.feats,
                                    n_resblocks=args.blocks, scale=args.scale,
                                    dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in generator.parameters())
    LOG.info('Generator: %.1fM params (dropout %.3f)', n_params / 1e6, args.dropout)

    discriminator = None
    d_optimizer = None
    if args.model == 'esrgan':
        discriminator = ESRGANDiscriminator(in_channels=n_channels).to(device)
        d_n_params = sum(p.numel() for p in discriminator.parameters())
        LOG.info('Discriminator: %.1fM params', d_n_params / 1e6)
        d_optimizer = torch.optim.AdamW(discriminator.parameters(),
                                        lr=args.lr * 4, weight_decay=1e-4)
        for p in discriminator.parameters():
            p.requires_grad_(True)

    use_perceptual = args.perceptual > 0 and n_channels == 3
    perceptual = PerceptualLoss(device) if use_perceptual else None
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()
    g_gan_loss = nn.BCEWithLogitsLoss()

    hparams = {
        'model_type': args.model,
        'n_colors': n_channels,
        'n_feats': args.feats,
        'n_resblocks': args.blocks,
        'scale': args.scale,
        'patch': args.patch,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'dropout': args.dropout,
    }

    if not use_perceptual:
        LOG.info('Perceptual loss disabled (requires exactly 3 bands)')
    if args.spectral > 0 and n_channels < 2:
        LOG.info('Spectral loss disabled (requires >=2 bands)')

    best_psnr = -1.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / 'satelite_sr_best.pt'
    latest_path = out_dir / 'satelite_sr.pt'

    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        generator.train()
        if discriminator is not None:
            discriminator.train()
        running_loss = 0.0
        n_batches = 0

        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            fake = generator(lr)

            if discriminator is not None:
                real_label = torch.ones((lr.size(0), 1), device=device)
                fake_label = torch.zeros((lr.size(0), 1), device=device)

                # --- train discriminator ---
                d_optimizer.zero_grad()
                d_real = discriminator(hr)
                d_fake = discriminator(fake.detach())
                loss_d = (g_gan_loss(d_real, real_label) +
                          g_gan_loss(d_fake, fake_label)) / 2
                loss_d.backward()
                d_optimizer.step()

                # --- train generator ---
                optimizer.zero_grad()
                adv_loss = g_gan_loss(discriminator(fake), real_label)
                loss = criterion(fake, hr) + args.gan * adv_loss
            else:
                optimizer.zero_grad()
                loss = criterion(fake, hr)

            if use_perceptual:
                loss = loss + args.perceptual * perceptual(fake, hr)
            if args.spectral > 0 and n_channels >= 2:
                loss = loss + args.spectral * spectral_angle_loss(fake, hr)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        val_psnr, val_ssim = validate(generator, val_loader, device)
        metrics = {'psnr': val_psnr, 'ssim': val_ssim}
        LOG.info('Epoch %3d/%d  loss %.4f  val_PSNR %.2f  val_SSIM %.4f',
                 epoch, args.epochs, running_loss / max(n_batches, 1),
                 val_psnr, val_ssim)

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            no_improve = 0
            save_checkpoint(best_path,
                            generator.state_dict() if discriminator is None else
                            {'generator': generator.state_dict(),
                             'discriminator': discriminator.state_dict()},
                            optimizer, scheduler, epoch, hparams, metrics,
                            is_best=True)
        else:
            no_improve += 1
            LOG.info('No PSNR improvement for %d epoch(s) (best %.2f)',
                     no_improve, best_psnr)
        if epoch % 5 == 0 or epoch == args.epochs:
            save_checkpoint(latest_path,
                            generator.state_dict() if discriminator is None else
                            {'generator': generator.state_dict(),
                             'discriminator': discriminator.state_dict()},
                            optimizer, scheduler, epoch, hparams, metrics)

        if no_improve >= args.patience:
            LOG.info('Early stopping at epoch %d (best PSNR %.2f)', epoch, best_psnr)
            break

    LOG.info('Done. Best PSNR %.2f dB saved to %s', best_psnr, best_path)


if __name__ == '__main__':
    main()