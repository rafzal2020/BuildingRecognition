import os

from flask import Flask, request, jsonify
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import wikipediaapi
import requests
from PIL import Image
import numpy as np
import pandas as pd
import fetch_image
from flask_cors import CORS
from wikipedia_scraper import get_infobox_fields
from fetch_image import get_wikidata_image


tf.disable_eager_execution()

# load model
model_url = 'https://www.kaggle.com/models/google/landmarks/TensorFlow1/classifier-north-america-v1/1'
m = hub.load(model_url)


def load_class_names(file_path):
    df = pd.read_csv(file_path)
    return dict(zip(df['id'].astype(str), df['name']))


class_names = load_class_names('landmarks_north_america.csv')


def preprocess_image(image_path):
    img = Image.open(image_path).resize((321, 321))
    img = np.array(img) / 255.0
    return img


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

global_name = ""
def get_wikipedia_info(building_name):
    user_agent = "BuildingRecognitionApp/1.0 (contact: rafzal2014@gmail.com)"
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent)

    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={building_name}&format=json"
    try:
        search_response = requests.get(search_url, headers={"User-Agent": user_agent}, timeout=5).json()
        if "query" in search_response and search_response["query"]["search"]:
            best_match_title = search_response["query"]["search"][0]["title"]  # Get the most relevant result
        else:
            return {"description": "No information available.", "wikipedia_link": None}
    except requests.RequestException:
        return {"description": "No information available.", "wikipedia_link": None}

    page = wiki_wiki.page(best_match_title)

    if not page.exists():
        return {"description": "No information available.", "wikipedia_link": None}

    description = page.summary.split("\n")[0]
    wikipedia_link = page.fullurl
    print(best_match_title)

    image_url = get_wikidata_image(best_match_title)
    print("image_url", image_url)

    return {
        "description": description,
        "wikipedia_link": wikipedia_link,
        "image_url": image_url
    }


# Create a Flask application
app = Flask(__name__)
# Update CORS configuration to be more specific
CORS(app, resources={
    r"/predict": {
        "origins": ["https://buildingrecognition.onrender.com"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/predict', methods=['POST'])
def upload_image():
    try:
        print("Request received")
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files['image']

        if not image.filename:
            return jsonify({"error": "No selected file"}), 400

        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        # Pass uploaded image to the model
        predicted_label_index, predicted_building_name = predict(image_path)
        print(f"Predicted building name: {predicted_building_name}")  # Debug print
        

        wiki_info = get_wikipedia_info(predicted_building_name)
        print("1")
        print(global_name)
        
        try:
            building_info = get_infobox_fields(predicted_building_name, ["architectural", "floor count", "status", "completed", "topped-out", "location"])
        except ValueError as e:
            print(f"Could not find building info: {str(e)}")
            building_info = {
                "architectural": None,
                "floor count": None,
                "status": None,
                "completed": None,
                "topped-out": None,
                "location": None,
            }

    

        return jsonify({
            "building_name": predicted_building_name,
            'confidence': 0.95,
            "wikipedia_info": wiki_info,
            "height": building_info["architectural"],
            "floors": building_info["floor count"],
            "status": building_info["status"],
            "completed": building_info["completed"],
            "topped-out": building_info["topped-out"],
            "location": building_info["location"],
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500


if __name__ == '__main__':
    app.run(debug=True)
