import os
import pickle
import time
from dotenv import load_dotenv
import streamlit as st

# Prevent Windows issues with torch
os.environ['TORCH_DISABLE_CLASSES'] = '1'

# Hugging Face & LangChain imports
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers.pipelines import pipeline
import torch

# Load environment variables (e.g., HF_TOKEN if needed)
load_dotenv()

# Streamlit UI
st.title("QuickNews: News Research Tool 📈")
st.sidebar.title("News Article URLs")

st.info("""
**How to use this tool:**
1. Enter news article URLs in the sidebar  
2. Click "Process URLs" to analyze the articles  
3. Ask a question based on those articles  
4. Get a friendly, sourced answer 💡

📝 This tool only answers questions based on the URLs you provide.
""")

# Input: URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_generative.pkl"
main_placeholder = st.empty()

# Load a text-to-text generative model
qa_model_name = "google/flan-t5-large"
device = 0 if torch.cuda.is_available() else "cpu"

text_gen_pipeline = pipeline(
    "text2text-generation",
    model=qa_model_name,
    tokenizer=qa_model_name,
    device=device
)

# Helper to safely get first output from pipeline call (list, generator, or else)
def get_first_output(output):
    if hasattr(output, '__iter__') and not isinstance(output, (str, bytes)):
        try:
            # Try treating as generator or iterator
            return next(output)
        except TypeError:
            # Probably a list or similar
            try:
                return output[0]
            except (IndexError, TypeError):
                return output
    return output

# Handle URL processing
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("🔄 Loading articles...")
    data = loader.load()

    # Track sources in metadata
    for i, doc in enumerate(data):
        if i < len(urls):
            doc.metadata["source"] = urls[i]

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=500
    )
    main_placeholder.text("🔍 Splitting text into chunks...")
    docs = text_splitter.split_documents(data)

    # Build vector index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("📦 Building search index...")
    time.sleep(1)

    # Save to disk
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    main_placeholder.success("✅ Articles processed!")

# Handle user query
query = main_placeholder.text_input("Ask a question based on the above articles:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        try:
            # Retrieve relevant docs
            docs = vectorstore.similarity_search(query, k=5)

            best_answer = None
            best_score = -1
            best_doc = None

            for doc in docs:
                prompt = f"""You are a helpful assistant. Based on the article below, answer the question in a complete sentence.

Article:
{doc.page_content}

Question: {query}
Answer:"""

                raw_output = text_gen_pipeline(prompt, max_new_tokens=200)
                first_output = get_first_output(raw_output)
                result = first_output.get("generated_text") if isinstance(first_output, dict) else str(first_output)

                if result and (len(result) > best_score):
                    best_score = len(result)
                    best_answer = result
                    best_doc = doc

            # Display answer
            st.header("Answer")
            st.write(str(best_answer).strip())

            # Display used source
            st.subheader("Source:")
            if best_doc and "source" in best_doc.metadata:
                st.write(best_doc.metadata["source"])
            else:
                st.write("No source URL available.")

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            st.info("Try reloading the articles or asking a simpler question.")
