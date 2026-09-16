import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image


class ResidualBlock(nn.Module):
    """
    A standard EDSR style residual block that removes batch normalization 
    layers to save memory and stabilize training performance.
    """
    def __init__(self, n_feats=256, kernel_size=3, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, n_colors=1, n_feats=256, n_resblocks=32, scale=4):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(n_colors, n_feats, 3, padding=1)

        self.body = nn.Sequential(*[
            ResidualBlock(n_feats) for _ in range(n_resblocks)
        ])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale // 2), 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, n_feats * (scale // 2), 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, n_colors, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = self.body_tail(res)
        x = x + res
        x = self.upsampler(x)
        return x


class IRSuperResolution:
    def __init__(self, scale=4, device=None):
        self.scale = scale
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = EDSR(n_colors=1, n_feats=256, n_resblocks=32, scale=self.scale)
            state = torch.load(
                'checkpoints/edsr_base_4x.pt',
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
        except (FileNotFoundError, RuntimeError):
            self.model = None

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.medianBlur(image, 3)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return enhanced

    def enhance(self, image: np.ndarray) -> np.ndarray:
        enhanced = self.preprocess(image)
        if self.model is not None:
            enhanced = self._super_resolve(enhanced)
        else:
            h, w = enhanced.shape[:2]
            enhanced = cv2.resize(enhanced, (w * self.scale, h * self.scale),
                                  interpolation=cv2.INTER_CUBIC)
            enhanced = cv2.detailEnhance(
                cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                sigma_s=10, sigma_r=0.15
            )
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        return enhanced

    def _super_resolve(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')

        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        tensor = tensor.view(1, 1, *tensor.shape).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        output = output.squeeze().cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        out_h = h * self.scale
        out_w = w * self.scale
        output = output[:out_h, :out_w]
        return output

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.enhance(image)


class SuperResolutionEDSR(nn.Module):
    """EDSR variant that supports configurable band count and integer scales (2/3/4).

    Used for multi-band (e.g. 4-band RGBNIR) satellite super-resolution. The
    upsampler uses a single PixelShuffle step so scale=4 → 2.5 m (from 10 m
    input) and scale=3 → 3.33 m, both satisfying <4 m output.
    """

    def __init__(self, n_colors=4, n_feats=256, n_resblocks=32, scale=4,
                 dropout=0.0):
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError('integer scale must be 2, 3 or 4')
        self.scale = scale
        self.n_colors = n_colors
        self.n_feats = n_feats
        self.n_resblocks = n_resblocks
        self.dropout = dropout

        self.head = nn.Conv2d(n_colors, n_feats, 3, padding=1)
        blocks = []
        for _ in range(n_resblocks):
            blocks.append(ResidualBlock(n_feats))
            if dropout > 0:
                blocks.append(nn.Dropout2d(dropout))
        self.body = nn.Sequential(*blocks)
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, n_colors, 3, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = self.body_tail(res)
        x = x + res
        x = self.upsampler(x)
        return x


class SatelliteSuperResolution:
    """Multi-band super-resolution wrapper for a trained SuperResolutionEDSR.

    Loads ``checkpoints/satelite_sr.pt`` (produced by ``training/train.py``)
    and enhances single- or multi-channel satellite images while preserving
    band structure — no grayscale conversion, so spectral consistency is kept.
    """

    def __init__(self, scale=4, n_channels=4,
                 checkpoint='checkpoints/satelite_sr.pt', device=None):
        self.scale = scale
        self.n_channels = n_channels
        self.checkpoint = checkpoint
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()

    @staticmethod
    def _extract_state_dict(state):
        if isinstance(state, dict):
            for key in ('model', 'state_dict', 'generator'):
                if key in state and isinstance(state[key], dict):
                    return SatelliteSuperResolution._extract_state_dict(state[key])
            if any(k.startswith(('head.', 'body.', 'upsampler.')) for k in state):
                return state
        return None

    def _load_model(self):
        try:
            state = torch.load(self.checkpoint, map_location=self.device,
                               weights_only=False)
        except (FileNotFoundError, RuntimeError, AttributeError):
            self.model = None
            return

        hparams = state.get('hparams', {}) if isinstance(state, dict) else {}
        n_colors = int(hparams.get('n_colors', self.n_channels))
        self.scale = int(hparams.get('scale', self.scale))
        n_feats = int(hparams.get('n_feats', 256))
        n_resblocks = int(hparams.get('n_resblocks', 32))
        self.dropout = float(hparams.get('dropout', 0.0))
        self.model_type = str(hparams.get('model_type', 'edsr'))

        weights = self._extract_state_dict(state)
        if weights is None:
            self.model = None
            return

        self.model = SuperResolutionEDSR(
            n_colors=n_colors, n_feats=n_feats,
            n_resblocks=n_resblocks, scale=self.scale,
            dropout=self.dropout,
        )
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _to_float(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if np.issubdtype(arr.dtype, np.uint16):
            img = arr.astype(np.float32) / 65535.0
        elif np.issubdtype(arr.dtype, np.uint8):
            img = arr.astype(np.float32) / 255.0
        else:
            img = arr.astype(np.float32)
            if img.size and img.max() > 1.5:
                img = img / img.max()
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def _to_c_h_w(image: np.ndarray, n_channels: int) -> np.ndarray:
        if image.ndim == 2:
            channels = image[np.newaxis, ...]
        else:
            channels = np.moveaxis(image, -1, 0)
        if channels.shape[0] < n_channels:
            pad = np.zeros(
                (n_channels - channels.shape[0], *channels.shape[1:]),
                dtype=channels.dtype,
            )
            channels = np.concatenate([channels, pad], axis=0)
        elif channels.shape[0] > n_channels:
            channels = channels[:n_channels]
        return channels

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                f'No trained checkpoint found at {self.checkpoint}. '
                'Run training/train.py first.')
        n_channels = self.model.n_colors
        in_ndim = image.ndim
        img = self._to_float(image)
        img = self._to_c_h_w(img, n_channels)

        _, h, w = img.shape
        s = self.scale
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h or pad_w:
            img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        out = out.squeeze(0).cpu().numpy()
        out = out[:, :h * s, :w * s]

        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        out = np.moveaxis(out, 0, -1)  # (H, W, C)
        if in_ndim == 2:
            out = out[..., 0]
        return out

    def enhance_with_uncertainty(self, image: np.ndarray, n_samples: int = 10
                                 ) -> tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo dropout inference.

        Returns ``(enhanced_uint8, std_map)`` where ``std_map`` is the per-pixel
        per-band standard deviation in the ``[0, 1]`` domain. Requires a model
        trained with ``--dropout > 0``, otherwise the estimated uncertainty is
        meaningless.
        """
        if self.model is None:
            raise RuntimeError(
                f'No trained checkpoint found at {self.checkpoint}. '
                'Run training/train.py first.')
        if self.dropout <= 0:
            raise RuntimeError(
                'Uncertainty estimation requires a model trained with '
                '--dropout > 0 (MC-dropout). Retrain enabling dropout.')

        from models.uncertainty import estimate_uncertainty

        n_channels = self.model.n_colors
        in_ndim = image.ndim
        img = self._to_float(image)
        img = self._to_c_h_w(img, n_channels)

        _, h, w = img.shape
        s = self.scale
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h or pad_w:
            img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        mean, std = estimate_uncertainty(
            self.model, tensor, n_samples=n_samples, device=self.device)
        mean = mean.squeeze(0).cpu().numpy()[:, :h * s, :w * s]
        std = std.squeeze(0).cpu().numpy()[:, :h * s, :w * s]

        mean_u8 = np.clip(mean * 255.0, 0, 255).astype(np.uint8)
        mean_u8 = np.moveaxis(mean_u8, 0, -1)
        std = np.moveaxis(std, 0, -1)  # (H, W, C) float [0, 1]
        if in_ndim == 2:
            mean_u8 = mean_u8[..., 0]
            std = std[..., 0]
        return mean_u8, np.clip(std, 0.0, 1.0).astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.enhance(image)
