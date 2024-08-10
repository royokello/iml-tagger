import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

class ImageDataset(Dataset):
    def __init__(self, working_dir, transform=None):
        self.image_dir = os.path.join(working_dir, 'ranker', 'output', '256p')
        self.label_dir = os.path.join(working_dir, 'tagger', 'labels')
        self.transform = transform
        self.image_filenames = []

        self.labels = []
        self.unique_classes = set()

        for filename in os.listdir(self.label_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(self.label_dir, filename), 'r') as file:
                    label = file.read().strip()
                    tags = [tag.strip() for tag in label.split(',')]
                    self.labels.append(tags)
                    self.unique_classes.update(tags)
                    # replace '.txt' with '.png' and append to image_filenames

        self.tag_to_index = {tag: idx for idx, tag in enumerate(self.unique_classes)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_filename = os.listdir(self.image_dir)[idx]
        img_path = os.path.join(self.image_dir, image_filename)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = self.labels[idx]
        label_indices = torch.zeros(len(self.unique_classes))
        for tag in self.labels[idx]:
            label_indices[self.tag_to_index[tag]] = 1
        return image, label_indices

    def get_num_classes(self):
        return len(self.unique_classes)
    

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)  # assuming images are already tensors
    labels = pad_sequence(labels, batch_first=True, padding_value=0)  # Pad labels to the longest in the batch
    return images, labels