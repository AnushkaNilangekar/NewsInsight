# QuickNews: AI News Research Tool

QuickNews is a web app that helps you quickly extract insights and answers from news articles you provide. Using Hugging Face's powerful language models and LangChain, it loads articles from URLs, indexes the content, and lets you ask natural language questions to get fast, sourced, and friendly answers.

---

## Demo
https://www.loom.com/share/911a0155d1b84e7aa6174a6ff3a21e15

## How to Use

1. Run the app:  

2. Enter up to 3 news article URLs in the given boxes.

3. Click Process URLs to load and analyze the articles.

4. Ask your question in the question box and click "Ask Question".

5. See the answer and the source article link(s).


## Setup
- Install Python 3.8 or higher.
- Clone the repository:
   - git clone
   - cd quicknews

### Backend:
- cd backend
- Create and activate a virtual environment (recommended):
   - python -m venv venv
   - source venv/bin/activate   (On Windows use: venv\Scripts\activate)
- Install required packages by running: pip install -r requirements.txt
- Add your Hugging Face API token in a .env file if you use private models.
- Run python app.py

### Frontend:
- cd frontend
- npm install all the packages
- npm run dev


## Notes
- The app works best with clear, public news URLs.

- Answers come from the articles you provide.

- You don’t need a GPU; it works on CPU too.

## License
MIT License
