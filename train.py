
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset import ImageDataset, collate_fn
from model import TaggerTransformer
from utils import get_model_by_name, log_print, setup_logging


def train(working_dir: str, epochs: int, checkpoint: int, base_model: str|None):
    tagger_dir = os.path.join(working_dir, 'tagger')
    setup_logging(tagger_dir)
    log_print("training started ...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"using {device}")

    # Setup directories
    models_dir = os.path.join(working_dir, 'tagger', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    dataset = ImageDataset(working_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Initialize the model, optimizer, and loss function
    if base_model:
        model = get_model_by_name(device=device, directory=models_dir, name=base_model)
    else:
        model = TaggerTransformer(dataset.get_num_classes()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001) # type: ignore
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        log_print(f"epoch: {epoch+1}, loss: {total_loss / len(dataloader)}")

        # Save checkpoint
        if (epoch + 1) % checkpoint == 0:
            torch.save(model.state_dict(), os.path.join(models_dir, f'model_epoch_{epoch+1}.pth'))
            log_print(f"checkpoint saved at epoch {epoch+1}")

    log_print("training completed!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the training data is located.")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train.")
    parser.add_argument("-c", "--checkpoint", type=int, required=True, help="Number of checkpoints to save.")
    parser.add_argument("-b", "--base_model", type=str, help="Name of the base model if continuing training, or None if starting from scratch.")
    
    args = parser.parse_args()
    
    train(
        working_dir=args.working_dir,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
        base_model=args.base_model
    )