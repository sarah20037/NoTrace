import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage, ToTensor

totensor = ToTensor()
topil = ToPILImage()

def resize_and_crop(img, size, crop_type="center"):
    if crop_type == "top":
        center = (0, 0)
    elif crop_type == "center":
        center = (0.5, 0.5)
    else:
        raise ValueError("crop_type must be 'top' or 'center'")
    
    resize = list(size)
    if size[0] is None:
        resize[0] = img.size[0]
    if size[1] is None:
        resize[1] = img.size[1]
    return ImageOps.fit(img, resize, centering=center)



def recover_image(image, init_image, mask, background=False):
    image = totensor(image)

    if isinstance(mask, torch.Tensor):
        # ✅ Fix: remove extra dims before converting
        mask = topil(mask.squeeze(0).squeeze(0))

    mask = totensor(mask)[0]
    init_image = totensor(init_image)

    if background:
        image = image * (1 - mask) + init_image * mask

    return topil(image)



def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L")).astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)
    return mask, masked_image
