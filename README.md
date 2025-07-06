# Building Identifier App

A web application that lets you upload a photo of a building and returns the name of the building, along with a link to its Wikipedia page. This implementation uses **LangChain** and **OpenAI GPT**, to combine visual recognition with prompt-engineered language understanding.

---

## Demo


---

## Tech Stack

| Layer         | Tech                           |
|--------------|--------------------------------|
| Backend       | Python              |
| LLM           | LangChain + OpenAI GPT-3.5/GPT-4 |
| Frontend      | TypeScript |

---

## Features

- Upload an image of a building/skyscraper.
- Uses **Google Cloud Vision API** to detect the building in the image.
- Sends landmark info to **OpenAI GPT-4** via **LangChain** to get a detailed description.
- Returns the building name, brief information, and details such as height and year of completion.
- Easily extensible to include map integration or historical data.

---

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key
- pip (Python package manager)
- NextJS

---

### 📦 Installation

```bash
# Clone the repo
git clone https://github.com/rafzal2020/buildingrecognition.git
cd building-identifier

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
python app.py

# In a new terminal
cd frontend
npm install
npm run dev
