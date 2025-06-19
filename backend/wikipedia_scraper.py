import requests
from bs4 import BeautifulSoup
import wikipediaapi

# Define your user-agent (you can personalize it)
USER_AGENT = "BuildingRecognitionApp/1.0 (contact: rafzal2014@gmail.com)"
HEADERS = {"User-Agent": USER_AGENT}

def get_wikipedia_page(title):
    """Get full Wikipedia page URL and verify it exists"""
    wiki = wikipediaapi.Wikipedia("en", headers=HEADERS)
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={title}&format=json"
    try:
        search_response = requests.get(search_url, headers={"User-Agent": USER_AGENT}, timeout=5).json()
        if "query" in search_response and search_response["query"]["search"]:
            best_match_title = search_response["query"]["search"][0]["title"]  # Get the most relevant result
        else:
            return {"description": "No information available.", "wikipedia_link": None}
    except requests.RequestException:
        return {"description": "No information available.", "wikipedia_link": None}

    page = wiki.page(best_match_title)
    if not page.exists():
        raise ValueError("Wikipedia page not found.")
    return page.fullurl

def get_infobox_fields(title, fields_to_extract):
    """Scrape the infobox from the Wikipedia page using a custom User-Agent"""
    url = get_wikipedia_page(title)
    response = requests.get(url, headers=HEADERS, timeout=10)
    result = {field: None for field in fields} 

    if response.status_code != 200:
        raise ValueError("Failed to fetch the Wikipedia page.")

    soup = BeautifulSoup(response.text, "html.parser")
    infobox = soup.find("table", {"class": "infobox"})
    if not infobox:
        raise ValueError("Infobox not found.")
    if infobox:
            for row in infobox.find_all("tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    label = th.text.strip().lower()
                    value = td.text.strip()

                    for field in fields:
                        if field.lower() == label:
                            result[field] = value
                            break  # Stop checking once matched
    
    print(result)
    return result

# Fields you want to extract
fields = ["floor count", "status", "completed", "topped-out", "location", "antenna spire", "architectural"]

# Example usage
if __name__ == "__main__":
    building_name = "Key Tower"
    try:
        building_info = get_infobox_fields(building_name, fields)
        print(f"📘 Wikipedia data for {building_name}:\n")
        for key, val in building_info.items():
            print(f"{key.capitalize()}: {val}")
    except Exception as e:
        print("Error:", str(e))
