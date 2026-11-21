"""DataLoaders over the processed datasets (plan task D7.2)."""
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from cage.data.dataset import SceneDataset, ObjectShapeDataset, RandomFlip, collate_scenes

def make_loader(name, split, batch_size, classes=None, train=True, num_workers=8, extra=(), transform=None, limit=None):
    """extra: datasets of the same format concatenated with name. transform: applied to pix3d samples. limit: first N samples."""
    if name == "pix3d":
        ds, collate = ObjectShapeDataset(split, classes, transform=transform), None    # the default collate stacks equal shapes
    else:
        flip = RandomFlip() if train and not limit else None
        ds = SceneDataset(name, split, classes, transform=flip)
        if extra:
            ds = ConcatDataset([ds] + [SceneDataset(n, split, classes, transform=flip) for n in extra])
        collate = collate_scenes
    if limit:
        ds = Subset(ds, range(limit))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, collate_fn=collate,
                      num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

class MixedLoader:
    """Yields (name, batch). Each step draws one dataset with probability proportional to its weight."""
    def __init__(self, loaders, weights, steps, seed=0):
        self.loaders, self.steps, self.names = loaders, steps, list(loaders)
        self.p = torch.tensor([float(weights[n]) for n in self.names])
        self.g = torch.Generator().manual_seed(seed)

    def __iter__(self):
        iters = {n: iter(loader) for n, loader in self.loaders.items()}
        for _ in range(self.steps):
            n = self.names[torch.multinomial(self.p, 1, generator=self.g).item()]
            try:
                batch = next(iters[n])
            except StopIteration:                    # a small dataset restarts; the others continue
                iters[n] = iter(self.loaders[n])
                batch = next(iters[n])
            yield n, batch
