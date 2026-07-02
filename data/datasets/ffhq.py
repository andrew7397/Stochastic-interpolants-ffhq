import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FFHQDataset(Dataset):
    """
    Simple dataset for loading FFHQ images from a directory.
    Supports resizing and standard normalization.
    """

    def __init__(self, data_root, input_size=128, split="train"):
        self.data_root = data_root
        self.input_size = input_size
        self.split = split

        # Find all images
        self.image_paths = sorted(glob.glob(os.path.join(data_root, "*.png"))) + \
                           sorted(glob.glob(os.path.join(data_root, "*.jpg"))) + \
                           sorted(glob.glob(os.path.join(data_root, "*.jpeg")))
        
        # Filter for validity? Or just trust glob.
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_root}")

        # Basic transform: Resize -> Tensor -> Normalize
        # We resize to input_size (e.g. 128) to save memory/compute
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Map [0, 1] to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            path = self.image_paths[idx]
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            
            if self.transform:
                img = self.transform(img)

            return {
                "input_images": img,
                "data_id": os.path.basename(path)
            }
        except Exception as e:
            # If image is corrupt or missing, try a random one
            print(f"Warning: Error loading image {self.image_paths[idx]}: {e}. Skipping...")
            new_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_idx)
