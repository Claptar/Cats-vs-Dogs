import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image


class CatDogDataset(Dataset):
    def __init__(self, folder_path, transforms=None):
        super().__init__()
        self.folder_path = folder_path
        self.transforms = transforms
        self.files = [os.path.join(self.folder_path, file) for file in os.listdir(self.folder_path)]
        self.labels = [1 if 'dog' in filename else 0 for filename in self.files]
        self.annotation = pd.DataFrame({
            'file_path': self.files,
            'label': self.labels
        })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.annotation.file_path[idx]
        label = self.annotation.label[idx]

        image = Image.open(file)

        if self.transforms:
            image = self.transforms(image)

        return image, label


if __name__ == "__main__":
    dataset_path = '/home/user/PycharmProjects/Cats-vs-Dogs/data/train/'
    train_dataset = CatDogDataset(folder_path=dataset_path)
    print(f'Number of items = {len(train_dataset)}')
    image, label = train_dataset[10]
    image.show()
