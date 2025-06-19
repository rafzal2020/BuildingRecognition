import urllib.request
import json
import hashlib


def get_md5_checksum(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def format_filename(filename):
    return filename.replace(" ", "_")


def get_wikimedia_url(filename):
    formatted_filename = filename.replace(" ", "_")  # Replace spaces with underscores
    md5_hash = hashlib.md5(formatted_filename.encode()).hexdigest()  # Get MD5 checksum
    subfolder = f"{md5_hash[0]}/{md5_hash[:2]}"  # First char and first two chars for path

    url = f"https://upload.wikimedia.org/wikipedia/commons/{subfolder}/{formatted_filename}"
    return url


def get_wikidata_image(building_title):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&sites=enwiki&props=claims&titles={format_filename(building_title)}"

    # Fetch JSON data
    with urllib.request.urlopen(url) as response:
        data = json.load(response)

    # Extract the entity ID (QID)
    entities = data.get("entities", {})
    if not entities:
        return None  # No data found

    qid = next(iter(entities))  # Get the first QID dynamically

    # Extract the image filename from P18
    try:
        image_filename = entities[qid]["claims"]["P18"][0]["mainsnak"]["datavalue"]["value"]
        print(image_filename)
        return get_wikimedia_url(image_filename)
    except KeyError:
        return None  # No image available


# Example usage
building_title = "Willis Tower"
image_filename = get_wikidata_image(building_title)
print(image_filename)
