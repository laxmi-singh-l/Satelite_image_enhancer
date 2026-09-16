"""
Download pretrained model weights for the Satellite IR Enhancement system.

Usage:
    python download_models.py          # download all models
    python download_models.py --list   # list available models
    python download_models.py edsr     # download specific model
"""

import sys
import argparse
from pathlib import Path
import urllib.request
import hashlib


MODELS = {
    'edsr': {
        'url': 'https://github.com/sanghyun-son/EDSR-PyTorch/releases/download/v1.0/edsr_base_4x.pt',
        'path': 'checkpoints/edsr_base_4x.pt',
        'size_mb': 50,
        'description': 'EDSR 4× super-resolution for IR enhancement',
        'sha256': None,
    },
    'pix2pix_ir2rgb': {
        'url': None,
        'path': 'checkpoints/pix2pix_ir2rgb.pt',
        'size_mb': 84,
        'description': 'Pix2Pix UNet IR→RGB colorization (train from scratch or download custom)',
        'sha256': None,
    },
}


def get_size_str(size_mb: int) -> str:
    if size_mb >= 1000:
        return f'{size_mb / 1000:.1f} GB'
    return f'{size_mb} MB'


def download_file(url: str, dest: Path, expected_sha256: str | None = None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  Downloading {dest.name}...')
    try:
        urllib.request.urlretrieve(url, str(dest))
        if expected_sha256:
            sha = hashlib.sha256()
            with open(dest, 'rb') as f:
                while chunk := f.read(8192):
                    sha.update(chunk)
            if sha.hexdigest() != expected_sha256:
                print(f'  WARNING: SHA256 mismatch for {dest.name}')
                return False
        print(f'  Saved to {dest}')
        return True
    except Exception as e:
        print(f'  FAILED: {e}')
        return False


def list_models():
    print('\nAvailable models:')
    print(f'{"Name":<20} {"Size":<10} {"Status":<12} Description')
    print('-' * 80)
    for name, info in MODELS.items():
        path = Path(info['path'])
        status = '✓ downloaded' if path.exists() else '✗ not downloaded'
        size = get_size_str(info['size_mb'])
        print(f'{name:<20} {size:<10} {status:<12} {info["description"]}')
    print()


def download_model(name: str) -> bool:
    if name not in MODELS:
        print(f'Unknown model: {name}')
        print(f'Available: {", ".join(MODELS.keys())}')
        return False

    info = MODELS[name]
    dest = Path(info['path'])

    if dest.exists():
        print(f'{name} already exists at {dest}')
        return True

    if info['url'] is None:
        print(f'No download URL for {name}')
        print(f'  You need to train this model or download custom weights to {dest}')
        return False

    return download_file(info['url'], dest, info['sha256'])


def main():
    parser = argparse.ArgumentParser(description='Download model weights')
    parser.add_argument('model', nargs='?', help='Model name to download')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.model:
        success = download_model(args.model)
        sys.exit(0 if success else 1)
        return

    print('=' * 60)
    print('Satellite IR Enhancement - Model Downloader')
    print('=' * 60)

    all_ok = True
    for name in MODELS:
        if not download_model(name):
            all_ok = False

    if all_ok:
        print('\nAll models downloaded successfully!')
    else:
        print('\nSome models failed to download.')
        print('The system will use fallback methods for missing models.')

    list_models()


if __name__ == '__main__':
    main()
