import torch
from PIL import Image
from torchvision import transforms

from models import TaggerCNN

def predict_tags(device: torch.device, model: TaggerCNN, index_to_tag: dict[int, str], image_path: str) -> str:
    """
    Predicts tags for a given image using the trained model.
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device) # type: ignore

        with torch.no_grad():
            output = model(image_tensor)
            predicted_tags = (output > 0.5).squeeze().cpu().numpy()

        tags = [index_to_tag[i] for i in range(len(predicted_tags)) if predicted_tags[i]]

        return ", ".join(tags)

    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error predicting tags: {str(e)}")