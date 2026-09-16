import torch
import torch.nn as nn

import numpy as np
import cv2


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, activation='relu', dropout=0.0, norm=True):
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=4,
                stride=2 if down else 1,
                padding=1,
            )
        ]

        if norm:
            layers.append(nn.BatchNorm2d(out_ch))

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, n_filters=64):
        super().__init__()

        self.down1 = UNetBlock(in_channels, n_filters, down=True, activation='lrelu', norm=False)
        self.down2 = UNetBlock(n_filters, n_filters * 2, down=True, activation='lrelu')
        self.down3 = UNetBlock(n_filters * 2, n_filters * 4, down=True, activation='lrelu')
        self.down4 = UNetBlock(n_filters * 4, n_filters * 8, down=True, activation='lrelu')
        self.down5 = UNetBlock(n_filters * 8, n_filters * 8, down=True, activation='lrelu')
        self.down6 = UNetBlock(n_filters * 8, n_filters * 8, down=True, activation='lrelu')
        self.down7 = UNetBlock(n_filters * 8, n_filters * 8, down=True, activation='lrelu')
        self.down8 = UNetBlock(n_filters * 8, n_filters * 8, down=True, activation='lrelu', norm=False)

        self.up1 = nn.ConvTranspose2d(n_filters * 8, n_filters * 8, 4, stride=2, padding=1)
        self.up2 = UNetBlock(n_filters * 16, n_filters * 8, down=False, activation='relu', dropout=0.5)
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 8, 4, stride=2, padding=1)
        self.up4 = UNetBlock(n_filters * 16, n_filters * 8, down=False, activation='relu', dropout=0.5)
        self.up5 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 4, stride=2, padding=1)
        self.up6 = UNetBlock(n_filters * 8, n_filters * 4, down=False, activation='relu', dropout=0.5)
        self.up7 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, stride=2, padding=1)
        self.up8 = UNetBlock(n_filters * 4, n_filters * 2, down=False, activation='relu')
        self.up9 = nn.ConvTranspose2d(n_filters * 2, n_filters, 4, stride=2, padding=1)
        self.up10 = UNetBlock(n_filters * 2, n_filters, down=False, activation='relu')
        self.up11 = nn.ConvTranspose2d(n_filters, n_filters, 4, stride=2, padding=1)
        self.up12 = UNetBlock(n_filters * 2, n_filters, down=False, activation='relu')
        self.up13 = nn.ConvTranspose2d(n_filters, n_filters, 4, stride=2, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(n_filters, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], dim=1)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d6], dim=1)
        u4 = self.up4(u3)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d5], dim=1)
        u6 = self.up6(u5)
        u7 = self.up7(u6)
        u7 = torch.cat([u7, d4], dim=1)
        u8 = self.up8(u7)
        u9 = self.up9(u8)
        u9 = torch.cat([u9, d3], dim=1)
        u10 = self.up10(u9)
        u11 = self.up11(u10)
        u11 = torch.cat([u11, d2], dim=1)
        u12 = self.up12(u11)
        u13 = self.up13(u12)
        u13 = torch.cat([u13, d1], dim=1)
        return self.out(u13)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4, n_filters=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters * 4, n_filters * 8, 4, stride=1, padding=1),
            nn.BatchNorm2d(n_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


class IR2RGB:
    LANDSAT_PALETTE = {
        0: (60, 119, 181),    # Water - blue
        1: (34, 139, 34),      # Forest - forest green
        2: (154, 205, 50),     # Agriculture - yellow-green
        3: (178, 34, 34),      # Urban - firebrick red
        4: (210, 180, 140),    # Barren - tan
        5: (0, 206, 209),      # Wetland - dark turquoise
        6: (124, 252, 0),      # Grassland - lawn green
    }

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = None
        self._load_model()

    def _load_model(self):
        try:
            self.generator = UNetGenerator(in_channels=1, out_channels=3)
            state = torch.load(
                'checkpoints/pix2pix_ir2rgb.pt',
                map_location=self.device,
            )

            if isinstance(state, dict) and "state_dict" in state:
                self.generator.load_state_dict(state["state_dict"])
            else:
                self.generator.load_state_dict(state)

            self.generator.to(self.device)
            self.generator.eval()
        except (FileNotFoundError, RuntimeError, KeyError):
            self.generator = None

    def colorize(self, ir_image: np.ndarray, segmentation_mask: np.ndarray | None = None) -> np.ndarray:
        return self._infer(ir_image) if self.generator is not None else self._colorize_fallback(ir_image, segmentation_mask)

    def _infer(self, ir_image: np.ndarray) -> np.ndarray:
        h, w = ir_image.shape[:2]
        tensor = torch.from_numpy(ir_image.astype(np.float32) / 255.0)
        tensor = tensor.view(1, 1, h, w).to(self.device)

        if self.generator is None:
            raise RuntimeError("Generator model is not initialized")

        with torch.no_grad():
            output = self.generator(tensor)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = ((output + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return output

    @staticmethod
    def _colorize_fallback(ir_image: np.ndarray, seg_mask: np.ndarray | None = None) -> np.ndarray:
        ir_norm = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ir_3ch = cv2.applyColorMap(ir_norm, cv2.COLORMAP_INFERNO)

        if seg_mask is not None:
            overlay = np.zeros_like(ir_3ch)
            for class_id, color in IR2RGB.LANDSAT_PALETTE.items():
                mask = seg_mask == class_id
                overlay[mask] = color
            ir_3ch = cv2.addWeighted(ir_3ch, 0.4, overlay, 0.6, 0)

        return ir_3ch

    def __call__(self, ir_image: np.ndarray, seg_mask: np.ndarray | None = None) -> np.ndarray:
        return self.colorize(ir_image, seg_mask)
