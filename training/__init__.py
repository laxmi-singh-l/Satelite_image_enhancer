"""Paired medium/low-resolution → high-resolution satellite super-resolution
training and validation utilities.

- ``dataset``  — manifest CSV / folder loader, aligned LR-HR patch cropping,
  augmentation, and fixed train/val/test splits.
- ``train``    — CLI training loop (L1 + optional VGG perceptual loss, AdamW,
  cosine schedule, PSNR-based checkpointing).
- ``validate`` — CLI accuracy assessment on the held-out test split producing
  PSNR / SSIM / NRMSE / SAM / ERGAS / LPIPS metrics and a before-augmented
  comparison montage.
"""