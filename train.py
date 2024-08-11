
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset import ImageDataset
from models import TaggerCNN
from utils import generate_model_name, log_print, setup_logging


def main(working_dir: str, epochs: int, checkpoint: int, base_model: str|None):
    tagger_dir = os.path.join(working_dir, 'tagger')
    setup_logging(tagger_dir)
    log_print("training started ...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"using {device}")

    models_dir = os.path.join(working_dir, 'tagger', 'models')
    os.makedirs(models_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # loads 256p x 256p images and their captions 
    dataset = ImageDataset(working_dir, transform=transform)
    num_tags = dataset.get_num_tags()
    index_to_tag = {idx: tag for tag, idx in dataset.tag_to_index.items()}

    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Model initialization
    model = TaggerCNN(num_tags)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # type: ignore

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (images, tags) in dataloader:
            images = images.to(device)
            tags = tags.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        log_print(f"epoch: {epoch+1}, loss: {total_loss / len(dataloader)}")

        if (epoch + 1) % checkpoint == 0:
            checkpoint_name = generate_model_name(
                base_model=base_model,
                epochs=epoch + 1
            )
            checkpoint_path = os.path.join(models_dir, f"{checkpoint_name}.pth")
            torch.save(model.state_dict(), checkpoint_path)

            index_to_tag_path = os.path.join(models_dir, f"{checkpoint_name}.json")
            with open(index_to_tag_path, 'w') as f:
                json.dump(index_to_tag, f)

            log_print(f"checkpoint and tags saved for {checkpoint_name}.")

    log_print("training completed!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the training data is located.")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train.")
    parser.add_argument("-c", "--checkpoint", type=int, required=True, help="Number of checkpoints to save.")
    parser.add_argument("-b", "--base_model", type=str, help="Name of the base model if continuing training, or None if starting from scratch.")
    
    args = parser.parse_args()
    
    main(
        working_dir=args.working_dir,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
        base_model=args.base_model
    )