import os
import pickle
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers.pipelines import pipeline

# Load environment variables
load_dotenv()

# Prevent Windows issues with torch
os.environ['TORCH_DISABLE_CLASSES'] = '1'

# Initialize FastAPI app
app = FastAPI(
    title="QuickNews API",
    description="AI-powered news research tool API",
    version="2.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
VECTORSTORE_PATH = "faiss_store_generative.pkl"
QA_MODEL_NAME = "google/flan-t5-large"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = 0 if torch.cuda.is_available() else -1

# Global variables for models
text_gen_pipeline = None
embeddings = None

# Pydantic models for request/response
class URLRequest(BaseModel):
    urls: List[str]

class QuestionRequest(BaseModel):
    question: str

class ProcessResponse(BaseModel):
    status: str
    message: str
    chunks_processed: Optional[int] = None

class AnswerResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    sources: Optional[List[str]] = None
    message: Optional[str] = None

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    global text_gen_pipeline, embeddings
    
    try:
        print("Loading models...")
        
        # Load text generation model
        text_gen_pipeline = pipeline(
            "text2text-generation",
            model=QA_MODEL_NAME,
            tokenizer=QA_MODEL_NAME,
            device=DEVICE
        )
        
        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda" if DEVICE == 0 else "cpu"}
        )
        
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {str(e)}")

# Helper function to safely get first output from pipeline
def get_first_output(output):
    if hasattr(output, '__iter__') and not isinstance(output, (str, bytes)):
        try:
            return next(output)
        except TypeError:
            try:
                return output[0]
            except (IndexError, TypeError):
                return output
    return output

@app.get("/")
async def root():
    return {
        "message": "QuickNews API",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": text_gen_pipeline is not None and embeddings is not None
    }

@app.post("/api/process", response_model=ProcessResponse)
async def process_urls(request: URLRequest):
    """Process news article URLs and create vector store"""
    
    if not request.urls or all(not url.strip() for url in request.urls):
        raise HTTPException(status_code=400, detail="No valid URLs provided")
    
    try:
        # Filter out empty URLs
        valid_urls = [url.strip() for url in request.urls if url.strip()]
        
        # Load articles
        loader = UnstructuredURLLoader(urls=valid_urls)
        data = loader.load()
        
        # Add source metadata
        for i, doc in enumerate(data):
            if i < len(valid_urls):
                doc.metadata["source"] = valid_urls[i]
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=500
        )
        docs = text_splitter.split_documents(data)
        
        # Create vector store
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Save to disk
        with open(VECTORSTORE_PATH, "wb") as f:
            pickle.dump(vectorstore, f)
        
        return ProcessResponse(
            status="success",
            message=f"Successfully processed {len(valid_urls)} article(s)",
            chunks_processed=len(docs)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URLs: {str(e)}")

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question based on processed articles"""
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="No question provided")
    
    # Check if vectorstore exists
    if not os.path.exists(VECTORSTORE_PATH):
        return AnswerResponse(
            status="error",
            message="No processed documents found. Please process URLs first."
        )
    
    try:
        # Load vectorstore
        with open(VECTORSTORE_PATH, "rb") as f:
            vectorstore = pickle.load(f)
        
        # Get relevant documents
        docs = vectorstore.similarity_search(request.question, k=5)
        
        if not docs:
            return AnswerResponse(
                status="error",
                message="No relevant information found in the processed articles."
            )
        
        # Generate answer using the best matching document
        best_answer = None
        best_score = -1
        best_doc = None
        
        for doc in docs:
            prompt = f"""You are a helpful assistant. Based on the article below, answer the question in a complete sentence.

Article:
{doc.page_content}

Question: {request.question}
Answer:"""
            
            raw_output = text_gen_pipeline(prompt, max_new_tokens=200)
            first_output = get_first_output(raw_output)
            result = first_output.get("generated_text") if isinstance(first_output, dict) else str(first_output)
            
            if result and len(result) > best_score:
                best_score = len(result)
                best_answer = result
                best_doc = doc
        
        # Get unique sources
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs if "source" in doc.metadata]))
        
        return AnswerResponse(
            status="success",
            answer=best_answer.strip() if best_answer else "Unable to generate an answer.",
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)