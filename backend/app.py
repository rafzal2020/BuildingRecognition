import os

from flask import Flask, request, jsonify
import wikipediaapi
import requests
from PIL import Image
import fetch_image
from flask_cors import CORS
from wikipedia_scraper import get_infobox_fields
from fetch_image import get_wikidata_image
import openai
import base64
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
openai.api_key = os.getenv("OPENAI_API_KEY")

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


app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": ["http://localhost:3000"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/predict', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files['image']

        if not image.filename:
            return jsonify({"error": "No selected file"}), 400

        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Identify the building in the image only. Do not provide any additional information or punctuation. If you cannnot identify the building, respond with 'Unknown'."},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_base64}}
                ]}
            ],
            max_tokens=500,
        )
        os.remove(image_path)  # Clean up the uploaded image after processing
        if response.choices[0].message.content.strip() == "Unknown":
            return jsonify({
                "building_name": "Unknown",
                "confidence": 0.0,
                "wikipedia_info": {"description": "No information available.", "wikipedia_link": None, "image_url": None},
                "height": None,
                "floors": None,
                "status": None,
                "completed": None,
                "topped-out": None,
                "location": None,
            })
        else:
            predicted_building_name = response.choices[0].message.content.strip()
            print(f"Predicted building name: {predicted_building_name}")  # Debug print

        wiki_info = get_wikipedia_info(predicted_building_name)

        try:
            building_info = get_infobox_fields(predicted_building_name,
                                               ["architectural", "floor count", "status", "completed", "topped-out",
                                                "location"])
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
    port = int(os.environ.get("PORT", 5000))  # Use env PORT or fallback
    app.run(host="0.0.0.0", port=port, debug=True)
