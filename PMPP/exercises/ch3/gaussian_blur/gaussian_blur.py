from pathlib import Path

import torch
from skimage import data
from torchvision.io import write_png


def compile_extension():
    metal_source = (Path(__file__).parent / "gaussianBlur.metal").read_text()
    return torch.mps.compile_shader(metal_source)

def gaussian_blur(lib, x, blur_size, sigma=1.0):
    assert x.device.type == "mps"
    assert x.dtype == torch.uint8
    assert blur_size % 2 == 1

    channels, height, width = x.shape
    x_out = torch.empty_like(x)
    lib.gaussianBlur(x, x_out, blur_size, height, width, channels, sigma, threads=x.numel(), group_size=256)
    return x_out


def main():
    current_dir = Path(__file__).parent

    x = torch.from_numpy(data.astronaut()).permute(2, 0, 1).contiguous().to("mps")

    print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)
    print("size:", x.numel())

    ext = compile_extension()
    blur_size = 7

    y = gaussian_blur(ext, x, blur_size, sigma=2.0)

    print("Converted:")
    print("mean:", y.float().mean())
    print("Input image:", y.shape, y.dtype)

    save_path = current_dir / "output.png"
    print(f"Blurred the image. It is saved at {save_path}")
    write_png(y.cpu(), save_path)


if __name__ == "__main__":
    main()