from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
load_dotenv()
import os
import time

app = FastAPI()

class VideoURL(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    url: str
    question: str

def extract_video_id(url: str) -> str:
    try:
        parsed_url = urlparse(url)
        if (parsed_url.netloc == 'youtu.be'):
            return parsed_url.path[1:]
        if (parsed_url.netloc == 'www.youtube.com'):
            return parse_qs(parsed_url.query)['v'][0]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

def check_embeddings_exist(index_name: str) -> bool:
    try:
        pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT')
        )
        indexes = pc.list_indexes()
        return index_name in [index.name for index in indexes]
    except Exception as e:
        print(f"Error checking Pinecone index: {e}")
        return False

def sanitize_index_name(video_id: str) -> str:
    """Sanitize the video ID to create a valid Pinecone index name."""
    # Replace any non-alphanumeric characters with '-'
    sanitized = ''.join(c if c.isalnum() else '-' for c in video_id.lower())
    # Ensure name starts with a letter (Pinecone requirement)
    if not sanitized[0].isalpha():
        sanitized = 'v-' + sanitized
    # Limit length to 45 characters (Pinecone limit)
    return sanitized[:45]

class TranscriptError(Exception):
    """Custom exception for transcript processing errors"""
    pass

def process_video_embeddings(video_id: str):
    index_name = sanitize_index_name(video_id)
    
    pc = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENVIRONMENT')
    )
    
    # Initialize embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Check if index exists
    if check_embeddings_exist(index_name):
        print("Index exists, loading existing embeddings...")
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embedding_function,
            pinecone_api_key=os.getenv('PINECONE_API_KEY')
        )
    else:
        print("Creating new embeddings...")
        try:
            # Get transcript with proper error handling
            transcript_text = ""
            try:
                print(f"Fetching transcript for video {video_id}...")
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                
                # Validate transcript data
                if isinstance(transcript, list) and len(transcript) > 0:
                    # Make sure each item has a 'text' field
                    transcript_parts = []
                    for item in transcript:
                        if isinstance(item, dict) and 'text' in item:
                            transcript_parts.append(item['text'])
                    transcript_text = " ".join(transcript_parts)
                else:
                    raise TranscriptError("Invalid transcript format")
                
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                print(f"English transcript not found, trying other languages...")
                # Try getting transcript in other languages and translate
                try:
                    available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                    auto_transcript = available_transcripts.find_transcript(['en-US', 'en-GB', 'auto'])
                    if not auto_transcript:
                        raise TranscriptError("No suitable transcript found")
                    
                    translated = auto_transcript.translate('en')
                    transcript_list = translated.fetch()
                    
                    # Validate translated transcript
                    if isinstance(transcript_list, list) and len(transcript_list) > 0:
                        transcript_parts = []
                        for item in transcript_list:
                            if isinstance(item, dict) and 'text' in item:
                                transcript_parts.append(item['text'])
                        transcript_text = " ".join(transcript_parts)
                    else:
                        raise TranscriptError("Invalid translated transcript format")
                        
                except Exception as translation_error:
                    print(f"Translation error: {str(translation_error)}")
                    raise TranscriptError(f"Failed to get transcript: {str(translation_error)}")
            
            if not transcript_text.strip():
                raise TranscriptError("Empty transcript received")
            
            print(f"Successfully processed transcript, length: {len(transcript_text)}")
            
            # Create text chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.create_documents([transcript_text])
            
            if not chunks:
                raise TranscriptError("Failed to create text chunks")
            
            # Create new index with correct region for free tier
            try:
                print("Creating Pinecone index...")
                pc.create_index(
                    name=index_name,
                    dimension=384,  # dimension for all-MiniLM-L6-v2
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                print(f"Waiting for index '{index_name}' to be ready...")
                while not check_embeddings_exist(index_name):
                    time.sleep(1)
                
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"Index creation error: {str(e)}")
                    raise
                print("Index already exists, continuing...")
            
            vector_store = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embedding_function,
                index_name=index_name,
                pinecone_api_key=os.getenv('PINECONE_API_KEY')
            )
            
        except TranscriptError as e:
            print(f"Transcript error: {str(e)}")
            return None
        except Exception as e:
            print(f"An error occurred: {type(e).__name__}: {str(e)}")
            return None
    
    return vector_store

@app.post("/process-video")
async def process_video(video_request: VideoURL):
    try:
        video_id = extract_video_id(video_request.url)
        vector_store = process_video_embeddings(video_id)
        if vector_store:
            return {"status": "success", "message": "Video processed successfully"}
        return {
            "status": "error",
            "message": "Failed to process video. Please ensure the video has captions available."
        }
    except Exception as e:
        error_message = f"Error: {type(e).__name__}: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/ask-question")
async def ask_question(question_request: QuestionRequest):
    try:
        video_id = extract_video_id(question_request.url)
        answer = get_answer_from_video(video_id, question_request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_answer_from_video(video_id: str, question: str):
    vector_store = process_video_embeddings(video_id)
    
    if not vector_store:
        return "Error: Could not process video"
        
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Get relevant documents
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Create prompt
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant. Answer the question based on the context provided.
        If the context does not contain the answer, say "I don't know".
        
        Context: {context}
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    # Generate final prompt
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    
    # Get response from LLM
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv('GROQ_API_KEY')
    )
    response = llm.invoke(final_prompt)
    
    return response.content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)