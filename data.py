"""Paired image/mask transforms for Oxford-IIIT Pet trimaps."""

from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class PetBinaryDataset(Dataset):
    def __init__(self, source, indices, size=128, augment=False):
        self.source, self.indices, self.size, self.augment = source, indices, size, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, trimap = self.source[self.indices[index]]
        image = F.resize(image, [self.size, self.size], InterpolationMode.BILINEAR)
        trimap = F.resize(trimap, [self.size, self.size], InterpolationMode.NEAREST)
        if self.augment and random.random() < 0.5:
            image, trimap = F.hflip(image), F.hflip(trimap)
        labels = F.pil_to_tensor(trimap).to(torch.int64)
        # Oxford trimap: 1=pet, 2=background, 3=ambiguous boundary.
        return F.to_tensor(image), (labels == 1).float(), (labels != 3).float()


def loaders(root: Path, batch_size=8, size=128, seed=42, validation_size=500, workers=0):
    source = datasets.OxfordIIITPet(root, split='trainval', target_types='segmentation',
                                    download=True)
    test_source = datasets.OxfordIIITPet(root, split='test', target_types='segmentation',
                                         download=True)
    if not 0 < validation_size < len(source):
        raise ValueError('invalid validation size')
    order = torch.randperm(len(source), generator=torch.Generator().manual_seed(seed)).tolist()
    train = PetBinaryDataset(source, order[validation_size:], size, augment=True)
    validation = PetBinaryDataset(source, order[:validation_size], size)
    test = PetBinaryDataset(test_source, list(range(len(test_source))), size)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers),
            DataLoader(validation, batch_size=batch_size, num_workers=workers),
            DataLoader(test, batch_size=batch_size, num_workers=workers))
