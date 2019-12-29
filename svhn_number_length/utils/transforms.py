import torch
from torchvision import transforms


class CropToBox(object):
    def __init__(self, expand=True):
        """Crop image to a rectangular bounding box

        Args:
            expand (bool): whether bounding box should be expanded by 30% (default: True)
        """
        self.expand = expand

    def __call__(self, sample):
        img, bboxes = sample["image"], sample["bboxes"]

        i, j, h, w = self._get_bbox(img, bboxes)
        img = transforms.transforms.F.resized_crop(img, i, j, h, w, (64, 64))
        bboxes = self._get_bboxes(bboxes, i, j, h, w)

        sample = {"image": img, "bboxes": bboxes}
        return sample

    def _get_bbox(self, img, bboxes):
        """Gets bounding box coordinates using an image's metadata

        Args:
            bboxes (dict): bounding boxes for each digit

        Returns:
            i (int): Upper pixel coordinate of the cropped image
            j (int): Left pixel coordinate of the cropped image
            h (int): Height of the cropped image
            w (int): Width of the cropped image
        """
        i = min(bboxes["top"])
        j = min(bboxes["left"])

        # bottom = top + height
        bottom = max(bboxes["top"][i] + bboxes["height"][i]
                     for i in range(len(bboxes["label"])))
        h = bottom - i

        # right = left + width
        right = max(bboxes["left"][idx] + bboxes["width"][idx]
                    for idx in range(len(bboxes["label"])))
        w = right - j

        if self.expand:
            # Expand up 15% or until we reach edge
            i = max(0, round(i - 0.15 * h))
            # Expand left 15% or until we reach edge
            j = max(0, round(j - 0.15 * w))

            # Expand down 15% or until we reach edge
            h = min(img.height, round(h * 1.3))
            # Expand right 15% or until we reach edge
            w = min(img.width, round(w * 1.3))

        return i, j, h, w

    def _get_bboxes(self, bboxes, i, j, h, w):
        """Updates individual bounding boxes for each digit
        Follows the following rules:
            preserve ratios of digit height / image height and digit weight / image weight
            preserve ratios of top / image height and left / image width
            (top, left) should be the distance to (i, j) on x and y axis respectively

        Args:
            bboxes (dict): original bounding boxes for each digit
            i (int): Upper pixel coordinate of the cropped image
            j (int): Left pixel coordinate of the cropped image
            h (int): Height of the cropped image
            w (int): Width of the cropped image

        Returns:
            bboxes (dict): bounding boxes in the resized image for each digit
        """
        for idx in range(len(bboxes["label"])):
            # preserve ratios of digit height / image height and digit width / image width
            bboxes["height"][idx] = int(bboxes["height"][idx] / h * 64)
            bboxes["width"][idx] = int(bboxes["width"][idx] / w * 64)

            # (i, j) have now become pixel (0, 0) after cropping
            # (top, left) is the distance between the image and pixel (0, 0)
            bboxes["top"][idx] -= i
            bboxes["left"][idx] -= j
            # preserve ratio of top / image height and left / image width
            # e.g. a pixel at the bottom right in the orig. image should now be at pixel (64, 64)
            bboxes["top"][idx] = int(bboxes["top"][idx] / h * 64)
            bboxes["left"][idx] = int(bboxes["left"][idx] / w * 64)

        return bboxes


class RandomCrop54(transforms.RandomCrop):
    def __init__(self):
        """Crop random 54x54 image"""
        super().__init__(54)

    def __call__(self, sample):
        """Crops image

        Args:
            sample (dict): contains image and bounding box metadata

        Returns:
            sample (dict): cropped image with updated bounding boxes
        """
        img, bboxes = sample["image"], sample["bboxes"]
        i, j, h, w = self.get_params(img, self.size)
        img = transforms.transforms.F.crop(img, i, j, h, w)

        for idx in range(len(bboxes["label"])):
            top, left = bboxes["top"][idx], bboxes["left"][idx]
            bottom, right = top + \
                bboxes["height"][idx], left + bboxes["width"][idx]

            # (i, j) have now become pixel (0, 0) after cropping
            # (top, left) is the distance between the image and pixel (0, 0) in the x & y axes
            new_top = min(max(0, top - i), h)
            new_left = min(max(0, left - j), w)
            new_bottom = min(max(0, bottom - i), h)
            new_right = min(max(0, right - j), w)

            new_height = new_bottom - new_top
            new_width = new_right - new_left

            bboxes["top"][idx] = new_top
            bboxes["left"][idx] = new_left
            bboxes["height"][idx] = new_height
            bboxes["width"][idx] = new_width

        return {"image": img, "bboxes": bboxes}


class ToTensor(transforms.ToTensor):
    def __call__(self, sample):
        """
        Args:
            sample (dict): contains image and bounding box metadata

        Returns:
            sample (dict): image as standardized tensor with original bounding boxes
        """
        img, bboxes = sample["image"], sample["bboxes"]
        img = super(ToTensor, self).__call__(img)

        return {"image": img, "bboxes": bboxes}


class Normalize(transforms.Normalize):
    def __call__(self, sample):
        """
        Args:
            sample (dict): contains image as tensor and bounding box metadata

        Returns:
            sample (dict): image as normalized tensor with original bounding boxes
        """
        img, bboxes = sample["image"], sample["bboxes"]
        img = super(Normalize, self).__call__(img)

        return {"image": img, "bboxes": bboxes}


class RandomGrayScale(transforms.RandomGrayscale):
    def __call__(self, sample):
        """
        Args:
            sample (dict): contains image and bounding box metadata

        Returns:
            sample (dict): grayscaled image with original bounding boxes
        """
        img, bboxes = sample["image"], sample["bboxes"]
        img = super(RandomGrayScale, self).__call__(img)

        return {"image": img, "bboxes": bboxes}


class RunModel(object):
    def __init__(self, model):
        """Returns image as output of model
        Useful for autoencoder

        Args:
            model (nn.Module): trained model with a __call__ function
        """
        self.model = model

    def __call__(self, sample):

        img, bboxes = sample["image"], sample["bboxes"]
        with torch.no_grad():
            img = img.unsqueeze(0)
            img = self.model(img)
            img = img.view(3, 54, 54)
        return {"image": img, "bboxes": bboxes}


def get(p, mean, std, model=None):
    """Returns a transform object for use with a dataloader

    Args:
        p (float): Probability of an image being converted to grayscale
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        model (nn.Module): Pretained model with a __call__ function for running RunModel transform (default: None)

    Returns:
        transform (torchvision.transforms): transform object
    """
    transform = transforms.Compose([
        CropToBox(),
        RandomCrop54(),
        RandomGrayScale(p),
        ToTensor(),
        Normalize(mean, std)
    ])
    if model is not None:
        transform.transforms.append(RunModel(model))
    return transform
