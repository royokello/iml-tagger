
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset import ImageDataset
from models import TaggerResNet
from utils import generate_model_name, log_print, setup_logging


def main(working_dir: str, epochs: int):
    tagger_dir = os.path.join(working_dir, 'tagger')
    setup_logging(tagger_dir)
    log_print("training started ...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"using {device}")

    models_dir = os.path.join(tagger_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(working_dir, transform=transform)
    num_tags = dataset.get_num_tags()
    index_to_tag = {idx: tag for tag, idx in dataset.tag_to_index.items()}

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = TaggerResNet(num_tags).train().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        log_print(f"epoch: {epoch+1}, loss: {total_loss / len(dataloader)}")

    checkpoint_name = generate_model_name(epochs=epochs)
    checkpoint_path = os.path.join(models_dir, f"{checkpoint_name}_model.pth")
    torch.save(model.state_dict(), checkpoint_path)

    index_to_tag_path = os.path.join(models_dir, f"{checkpoint_name}_vocab.json")
    with open(index_to_tag_path, 'w') as f:
        json.dump(index_to_tag, f)

    log_print(f"model and vocab saved for {checkpoint_name}.")

    log_print("training completed.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the training data is located.")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train.")

    args = parser.parse_args()
    
    main(
        working_dir=args.working_dir,
        epochs=args.epochs,
    )