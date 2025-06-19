# QuickNews: AI News Research Tool 📈

QuickNews is a web app that helps you quickly extract insights and answers from news articles you provide. Using Hugging Face's powerful language models and LangChain, it loads articles from URLs, indexes the content, and lets you ask natural language questions to get fast, sourced, and friendly answers.

---

## How to Use

1. Run the app:  
   streamlit run app.py
2. Enter up to 3 news article URLs in the sidebar.

3. Click Process URLs to load and analyze the articles.

4. Ask your question in the main box.

5. See the answer and the source article link.


## Setup
- Install Python 3.8 or higher.
- Clone the repository:
git clone https://github.com/yourusername/quicknews.git

cd quicknews

- Create and activate a virtual environment (recommended):

python -m venv venv

source venv/bin/activate   # On Windows use: venv\Scripts\activate
- Install required packages by running: pip install -r requirements.txt
- Add your Hugging Face API token in a .env file if you use private models.


## Notes
- The app works best with clear, public news URLs.

- Answers come from the articles you provide.

- You don’t need a GPU; it works on CPU too.

## License
MIT License
