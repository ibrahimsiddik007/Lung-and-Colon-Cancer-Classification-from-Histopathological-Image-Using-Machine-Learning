import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.optimizers import Adamax
from tqdm import tqdm
import warnings

# Ignore Warnings
warnings.filterwarnings("ignore")

# Load the model
model_path = 'Model.h5'
assert os.path.exists(model_path), f"Error: Model not found at {model_path}"
model = tf.keras.models.load_model(model_path)
model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define class labels with correct subfolder paths
class_labels = {
    'Colon_Adenocarcinoma': 'colon_image_sets/colon_aca',
    'Colon Benign': 'colon_image_sets/colon_n',
    'Lung Adenocarcinoma': 'lung_image_sets/lung_aca',
    'Lung Benign Tissue': 'lung_image_sets/lung_n',
    'Lung Squamous Cell Carcinoma': 'lung_image_sets/lung_scc'
}

# Function to process and predict an image
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    return list(class_labels.keys())[tf.argmax(score)]

# Traverse and process images
def traverse_and_test_images(root_dir):
    mismatched_summary = {class_label: 0 for class_label in class_labels}

    for class_label, subfolder in class_labels.items():
        class_dir = os.path.join(root_dir, subfolder)
        if not os.path.exists(class_dir):
            print(f"⚠ Warning: Folder not found -> {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        with tqdm(total=len(files), desc=f"Processing {class_label}", unit="file") as pbar:
            for file in files:
                image_path = os.path.join(class_dir, file)
                predicted_class = process_image(image_path)
                if predicted_class != class_label:
                    mismatched_summary[class_label] += 1
                pbar.update(1)

    print("\nTotal Mismatched Images Summarized:")
    for class_label, count in mismatched_summary.items():
        print(f"{class_label}: {count}")

# Run the function
root_directory = 'datasets/lung_colon_image_set/'
traverse_and_test_images(root_directory)
