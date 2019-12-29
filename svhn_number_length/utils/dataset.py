import os
import pickle

from PIL import Image
from torch.utils.data import Dataset


class SVHNDataset(Dataset):
    """SVHN Dataset"""

    def __init__(self, md, img_dir, transform=None):
        """
        Args:
            md (str, path-like): Path to the pkl file with metadata
            img_dir (str, path-like): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        with open(md, "rb") as f:
            self.md = pickle.load(f)

        # md.keys() are just ints in [0-len(md)]. Use tuple for efficiency
        self.md = tuple(self.md.values())
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.md)

    def __getitem__(self, idx):
        img = self.md[idx]
        path_to_img = os.path.join(self.img_dir, img["filename"])
        bboxes = img["metadata"]
        bboxes["length"] = sum(1 for i in range(len(bboxes["label"]))
                               # exclude digits completely cropped by transform(s)
                               if bboxes["width"][i] > 0 or bboxes["height"][i] > 0
                               )

        img = Image.open(path_to_img)
        sample = {"image": img, "bboxes": bboxes}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
