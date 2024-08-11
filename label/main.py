import argparse
import json
import os
from flask import Flask, jsonify, render_template, request, send_from_directory
import torch
from inference import predict_tags
from utils import get_model_by_latest

app = Flask(__name__)

low_res_dir = ""
std_img_dir = ""
labels_dir = ""
labeled_image_ids = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_mappings = {}

@app.route('/')
def index():
    global std_img_dir, labels_dir, labeled_image_ids
    total_images = sum(1 for entry in os.scandir(std_img_dir) if entry.is_file())
    return render_template('index.html', total_images=total_images, total_labels=len(labeled_image_ids))

@app.route('/image/<int:img_id>')
def get_image(img_id):
    global std_img_dir
    print(f"img_id: {img_id}")
    return send_from_directory(std_img_dir, f"{img_id}.png")

@app.route('/next_labeled_image', methods=['POST'])
def next_labeled_image():
    global labeled_image_ids
    data = request.json
    if data and 'id' in data and 'step' in data:
        starting_point = min(range(len(labeled_image_ids)), key=lambda i: abs(labeled_image_ids[i] - data['id']))
        next_index = starting_point + data['step']
        if next_index < 0:
            next_index = len(labeled_image_ids) - 1
        elif next_index >= len(labeled_image_ids):
            next_index = 0
        next_img_id = labeled_image_ids[next_index]
        labels_path = os.path.join(labels_dir, f"{next_img_id}.txt")
        try:
            with open(labels_path, 'r') as file:
                label = file.read().strip()
        except FileNotFoundError:
            label = ""
        return jsonify(img_id=next_img_id, label=label)
    else:
        return jsonify(img_id=1)

@app.route('/label', methods=['POST'])
def label_image():
    global labels_path, labeled_image_ids
    data = request.json
    if data and 'id' in data and 'caption' in data:
        image_id = int(data['id'])
        labels_path = os.path.join(labels_dir, f"{image_id}.txt")
        with open(labels_path, 'w') as file:
            file.write(data['caption'])
        if image_id not in labeled_image_ids:
            labeled_image_ids.append(image_id)
            labeled_image_ids = sorted(list(set(labeled_image_ids)))
        return jsonify(success=True)
    else:
        return jsonify(success=False)
    
@app.route('/predict', methods=['POST'])
def predict_caption():
    global low_res_dir, device, model, class_mappings
    data = request.json
    print(f"data: {data}")
    if data:
        image_path = os.path.join(low_res_dir, f"{data['image_id']}.png")
        prediction = predict_tags(
            device=device,
            model=model,
            image_path=image_path,
            index_to_tag=class_mappings
        )
        return jsonify(prediction=prediction)
    else:
        return jsonify(success=False)

def label(working_dir: str):
    global std_img_dir, labels_dir, labeled_image_ids, device, model, class_mappings, low_res_dir
    low_res_dir = os.path.join(working_dir, 'ranker', 'output', '256p')
    std_img_dir = os.path.join(working_dir, 'ranker', 'output', '512p')
    labels_dir = os.path.join(working_dir, 'tagger', 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    labeled_image_ids = sorted([int(filename[:-4]) for filename in os.listdir(labels_dir) if filename.endswith('.txt')])
    models_dir = os.path.join(working_dir, 'tagger', 'models')
    model_and_class_mappings = get_model_by_latest(device=device, directory=models_dir)
    if model_and_class_mappings:
        model, class_mappings = model_and_class_mappings
        model.eval()
    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IML Tagger Labeler")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Working Directory in ILM Format.")
    
    args = parser.parse_args()
    label(args.working_dir)