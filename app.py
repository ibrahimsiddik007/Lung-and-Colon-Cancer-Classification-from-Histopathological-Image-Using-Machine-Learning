from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax


# Initialize Flask app
app = Flask(__name__,template_folder='templates',static_folder='static')


# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = tf.keras.models.load_model('Model.h5')
model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define class labels
class_labels = ['Colon_Adenocarcinoma', 'Colon Benign', 'Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

# Function to process and predict an image
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_labels[tf.argmax(score)]
    return predicted_class

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Predict the class of the uploaded image
            predicted_class = process_image(file_path)

            # Render the result
            return render_template('index.html', uploaded_image=file.filename, prediction=predicted_class)

    return render_template('index.html', uploaded_image=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)