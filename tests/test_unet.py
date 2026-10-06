import torch
from PIL import Image

from data import PetBinaryDataset
from model import UNet
from train import segmentation_loss


def test_output_shape_and_backward():
    model = UNet(base=8)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 1, 64, 64)
    output.mean().backward()
    assert model.out.weight.grad is not None


def test_ambiguous_border_has_no_effect_on_loss():
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits)
    valid[:, :, 0, 0] = 0
    original = segmentation_loss(logits, target, valid, 'combined')
    logits[:, :, 0, 0] = 100
    target[:, :, 0, 0] = 1
    assert torch.allclose(segmentation_loss(logits, target, valid, 'combined'), original)


def test_image_and_mask_flip_together(monkeypatch):
    image = Image.new('RGB', (16, 16), 'black')
    trimap = Image.new('L', (16, 16), 2)
    for x in range(8):
        for y in range(16):
            image.putpixel((x, y), (255, 0, 0))
            trimap.putpixel((x, y), 1)
    monkeypatch.setattr('data.random.random', lambda: 0.0)
    pixels, target, valid = PetBinaryDataset([(image, trimap)], [0], 16, augment=True)[0]
    assert pixels[0, :, :8].sum() == 0
    assert pixels[0, :, 8:].sum() == 128
    assert target[0, :, :8].sum() == 0
    assert target[0, :, 8:].sum() == 128
    assert valid.sum() == 256
