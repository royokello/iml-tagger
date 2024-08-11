import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, working_dir, transform=None):
        self.image_dir = os.path.join(working_dir, 'ranker', 'output', '256p')
        self.label_dir = os.path.join(working_dir, 'tagger', 'labels')
        self.transform = transform
        self.image_filenames = []

        self.labels = []
        self.unique_tags = set()

        for filename in os.listdir(self.label_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(self.label_dir, filename), 'r') as file:
                    label = file.read().strip()
                    tags = [tag.strip() for tag in label.split(',')]
                    self.labels.append(tags)
                    self.unique_tags.update(tags)
                image_filename = filename.replace('.txt', '.png')
                self.image_filenames.append(image_filename)

        self.tag_to_index = {tag: idx for idx, tag in enumerate(self.unique_tags)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, image_filename)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_indices = torch.zeros(len(self.unique_tags))
        for tag in self.labels[idx]:
            label_indices[self.tag_to_index[tag]] = 1
        return image, label_indices

    def get_num_tags(self):
        return len(self.unique_tags)
