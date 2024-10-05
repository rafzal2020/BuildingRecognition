import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import os
import pandas as pd
from flask import Flask, request, render_template

# Disable eager execution for TensorFlow 1.x
tf.disable_eager_execution()

# Load the model from TensorFlow Hub
model_url = 'https://www.kaggle.com/models/google/landmarks/TensorFlow1/classifier-north-america-v1/1'
m = hub.load(model_url)

# Load class names from CSV
def load_class_names(file_path):
    df = pd.read_csv(file_path)  # Load the CSV into a DataFrame
    return dict(zip(df['id'].astype(str), df['name']))  # Create a dictionary from the DataFrame

# Load class names from the CSV file
class_names = load_class_names('landmarks_north_america.csv')

# Function to preprocess the image
def preprocess_image(image_path):
    # Load and resize the image
    img = Image.open(image_path).resize((321, 321))  # Adjust size according to model requirements
    img = np.array(img) / 255.0  # Normalize the image
    return img

# Function to make a prediction
def predict(image_path):
    with tf.Graph().as_default():
        image = preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Create a session and run the model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # Initialize global variables
            sess.run(tf.tables_initializer())  # Initialize the lookup tables

            # Get the model's signature for predictions
            model = m.signatures['default']  # Access the default signature

            # Pass the image to the model
            pred = model(tf.constant(image, dtype=tf.float32))

            # Print the output to inspect its structure
            print(pred)

            # Check the keys of the output
            print("Output keys:", pred.keys())

            # Evaluate the predictions inside the session
            if 'default' in pred:
                prediction_output = sess.run(pred['default'])  # Run the prediction to get output
                label_index = np.argmax(prediction_output, axis=-1)[0]  # Get the index of the predicted label
            else:
                # If 'default' is not in pred, you may need to check other keys
                prediction_output = sess.run(pred[list(pred.keys())[0]])  # Access the first key
                label_index = np.argmax(prediction_output, axis=-1)[0]  # Get the index of the predicted label

            building_name = class_names.get(str(label_index), "Unknown Building")
            return label_index, building_name  # Return the index of the predicted label

# Create a Flask application
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    # Save the uploaded image
    image_path = os.path.join('images', file.filename)
    file.save(image_path)

    # Make prediction
    predicted_label_index, predicted_building_name = predict(image_path)

    # Create a Wikipedia link (you can modify this logic to fit your needs)
    wikipedia_link = f"https://en.wikipedia.org/wiki/{predicted_building_name.replace(' ', '_')}"

    return render_template('index.html', building_name=predicted_building_name, wikipedia_link=wikipedia_link)

if __name__ == '__main__':
    app.run(debug=True)
