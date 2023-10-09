from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import albumentations as A
import numpy as np
from torchvision.transforms import ToTensor

class LEVIRDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.image_paths_A = glob.glob(root_path + "/A/*.png")
        self.image_paths_B = glob.glob(root_path + "/B/*.png")
        self.image_paths_labels = glob.glob(root_path + "/label/*.png")

        self.image_paths_A.sort()
        self.image_paths_B.sort()
        self.image_paths_labels.sort()

        print("Found", len(self.image_paths_A), "samples in", root_path)
        
        assert len(self.image_paths_A) == len(self.image_paths_B) == len(self.image_paths_labels), \
            "Number of images in each folder must be the same."

    def __len__(self):
        return len(self.image_paths_A)

    def __getitem__(self, index):
        image_A = np.array(Image.open(self.image_paths_A[index]))
        image_B = np.array(Image.open(self.image_paths_B[index]))
        image_labels = np.array(Image.open(self.image_paths_labels[index]))
        
        if self.transform is not None:
            transformed = self.transform(image=image_A, image0=image_B, image1=image_labels)
        
        toTorchTensor = ToTensor() # convert to tensor with pytorch function to get values in the [0, 1] range
        return toTorchTensor(transformed['image']), toTorchTensor(transformed['image0']), (toTorchTensor(transformed['image1']) > 0.5).float() # correctly disccretize labels

def get(base_path, batch_size):
    from google.colab import drive
    drive.mount('/content/drive')

    train_root = base_path + "/train"
    val_root = base_path + "/val"
    test_root = base_path + "/test"

    transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate()
        ], 
        additional_targets={'image0': 'image', 'image1': 'image'}
    )

    train_set = LEVIRDataset(root_path=train_root, transform=transform)
    val_set = LEVIRDataset(root_path=val_root, transform=transform)
    test_set = LEVIRDataset(root_path=test_root, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader