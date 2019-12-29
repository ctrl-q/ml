# %%
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

import torch


def get_weighted_sampler(dataset):
    """Creates a weighted sampler to oversample minority classes

    Args:
        dataset (SVHNDataset): SVHNDataset
    Returns:
        sampler (WeightedRandomSampler): a weighted sampler for DataLoader
        purposes.
    """
    # Get training targets
    if dataset.transform is None:
        targets = np.array([len(sample["metadata"]["label"])
                            for sample in dataset.md])
    else:
        targets = np.array([sample["bboxes"]["length"]
                            for _, sample in enumerate(dataset)])

    # Count the number of class instances
    classes, class_counts = np.unique(targets, return_counts=True)

    # Oversample the minority classes
    weight = 1. / class_counts
    samples_weight = weight[targets - 1]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler
