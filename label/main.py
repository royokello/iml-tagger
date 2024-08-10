import argparse
import json
import os
from flask import Flask, jsonify, render_template, request, send_from_directory
import torch

app = Flask(__name__)

std_img_dir = ""
labels_dir = ""
labeled_image_ids = []
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
# @app.route('/predict', methods=['POST'])
# def predict_choice():
#     global low_res_path, device, model
#     data = request.json
#     if data:
#         image_1_path = os.path.join(low_res_path, f"{data['img_1_id']}.png")
#         image_2_path = os.path.join(low_res_path, f"{data['img_2_id']}.png")
#         prediction = predict(device=device, model=model, image_1_path=image_1_path, image_2_path=image_2_path)
#         return jsonify(prediction=prediction)
#     else:
#         return jsonify(success=False, message="Invalid data"), 400

def label(working_dir: str):
    global std_img_dir, labels_dir, labeled_image_ids
    std_img_dir = os.path.join(working_dir, 'ranker', 'output', '512p')
    labels_dir = os.path.join(working_dir, 'tagger', 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    labeled_image_ids = sorted([int(filename[:-4]) for filename in os.listdir(labels_dir) if filename.endswith('.txt')])
    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IML Tagger Labeler")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Working Directory in ILM Format.")
    
    args = parser.parse_args()
    label(args.working_dir)