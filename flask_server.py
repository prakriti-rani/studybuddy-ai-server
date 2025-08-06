from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from flask_cors import CORS
from dotenv import load_dotenv

# Load API Key
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Simple rate limiting
import time
last_request_time = 0
MIN_REQUEST_INTERVAL = 2  # Minimum 2 seconds between requests

# Helper Functions
def get_pdf(pdfs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdfs:
        pdfreader = PdfReader(pdf)
        for page in pdfreader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create a FAISS vector store."""
    try:
        print(f"Creating embeddings for {len(text_chunks)} chunks...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Embeddings model loaded successfully")
        
        print("Creating FAISS vector store...")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        print("FAISS vector store created")
        
        print("Saving vector store to disk...")
        vector_store.save_local("faiss_index")
        print("Vector store saved successfully")
        
    except Exception as e:
        print(f"ERROR in get_vector_store: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

def get_chats():
    """Configure the AI chat model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context." Don't provide the wrong answer.
    You can create summaries, question-answer pairs, and personalized flashcards for uploaded documents.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    # Try gemini-1.5-flash first (higher quota), fallback to gemini-1.5-pro
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        print("Using gemini-1.5-flash model")
    except:
        try:
            model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
            print("Using gemini-1.5-pro model")
        except:
            # Final fallback
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            print("Using gemini-1.5-flash model")
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def query_vector_store(query):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Check if FAISS index exists
        if not os.path.exists("faiss_index"):
            raise FileNotFoundError("No document has been processed. Please upload a document first.")
        
        print(f"Loading FAISS index for query: {query[:50]}...")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(query)
        print(f"Found {len(docs)} relevant documents")
        
        if not docs:
            return "No relevant information found in the document for this query."
        
        print("Generating AI response...")
        chain = get_chats()
        
        # Add retry logic for rate limiting
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
                print("AI response generated successfully")
                return response["output_text"]
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10  # Wait 10, 20, 30 seconds
                        print(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("Max retries reached for rate limiting")
                        return "I'm experiencing high demand right now. Please try again in a few moments."
                else:
                    raise e
        
    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        raise e
    except Exception as e:
        print(f"Error in query_vector_store: {str(e)}")
        if "429" in str(e) or "quota" in str(e).lower():
            return "The AI service is currently experiencing high demand. Please try again in a few moments."
        raise Exception(f"Failed to process query: {str(e)}")

# Flask API Endpoints
@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is running."""
    return jsonify({
        "status": "running",
        "message": "StudyBuddy AI Server is online",
        "google_api_configured": bool(os.environ.get("GOOGLE_API_KEY"))
    })

@app.route("/upload", methods=["POST"])
def upload():
    try:
        print("=== Upload request received ===")
        print(f"Request headers: {dict(request.headers)}")
        print(f"Request files: {list(request.files.keys())}")
        
        if "file" not in request.files:
            print("ERROR: No file in request")
            return jsonify({"error": "No file uploaded"}), 400
        
        pdf_file = request.files["file"]
        print(f"Received file: {pdf_file.filename}")
        
        # Read file content to check size
        file_content = pdf_file.read()
        print(f"File size: {len(file_content)} bytes")
        pdf_file.seek(0)  # Reset file pointer after reading size
        
        if pdf_file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        if len(file_content) == 0:
            print("ERROR: Empty file")
            return jsonify({"error": "File is empty"}), 400
        
        # Process the file
        print("Extracting text from PDF...")
        text = get_pdf([pdf_file])  # Pass as a list
        print(f"Extracted text length: {len(text) if text else 0} characters")
        
        if text:
            print(f"First 200 characters of text: {text[:200]}...")
        
        if not text or len(text.strip()) == 0:
            print("ERROR: No text extracted from PDF")
            return jsonify({"error": "Failed to extract text from PDF or PDF is empty"}), 500

        print("Splitting text into chunks...")
        text_chunks = get_text_chunks(text)
        print(f"Created {len(text_chunks)} text chunks")
        
        if text_chunks:
            print(f"First chunk length: {len(text_chunks[0])}")
        
        if not text_chunks:
            print("ERROR: No text chunks created")
            return jsonify({"error": "Failed to split text into chunks"}), 500

        print("Creating vector store...")
        get_vector_store(text_chunks)
        print("Vector store created successfully")
        
        # Verify the FAISS index was created
        if os.path.exists("faiss_index"):
            print("✓ FAISS index created successfully")
            # List contents of faiss_index directory
            faiss_files = os.listdir("faiss_index")
            print(f"FAISS index contains: {faiss_files}")
        else:
            print("ERROR: FAISS index not found after creation")
            return jsonify({"error": "Failed to create vector store"}), 500

        print("=== Upload completed successfully ===")
        return jsonify({"message": "File processed successfully"}), 200

    except Exception as e:
        print(f"ERROR in upload: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500



@app.route("/ask", methods=["POST"])
def ask():
    """Endpoint for asking questions based on the document."""
    data = request.json
    user_question = data.get("question")
    
    if not user_question:
        return jsonify({"error": "Question is required"}), 400
    
    response = query_vector_store(user_question)
    return jsonify({"response": response})

@app.route("/summary", methods=["POST"])
def summary():
    """Generate a summary of the uploaded document."""
    global last_request_time
    
    try:
        print("=== Summary request received ===")
        
        # Simple rate limiting
        current_time = time.time()
        if current_time - last_request_time < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - (current_time - last_request_time)
            print(f"Rate limiting: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        last_request_time = time.time()
        
        # Check if FAISS index exists
        if not os.path.exists("faiss_index"):
            print("ERROR: FAISS index not found")
            print(f"Current directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            return jsonify({"error": "No document uploaded. Please upload a document first."}), 400
        
        print("FAISS index found, generating summary...")
        response = query_vector_store("Provide a comprehensive summary of the entire document, highlighting the main topics, key points, and important details.")
        print(f"Summary generated: {len(response)} characters")
        print("=== Summary completed successfully ===")
        return jsonify({"summary": response})
        
    except Exception as e:
        print(f"ERROR generating summary: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return jsonify({"error": "Failed to generate summary", "details": str(e)}), 500

@app.route("/flashcards", methods=["POST"])
def flashcards():
    global last_request_time
    
    try:
        print("=== Flashcards request received ===")
        
        # Simple rate limiting
        current_time = time.time()
        if current_time - last_request_time < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - (current_time - last_request_time)
            print(f"Rate limiting: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        last_request_time = time.time()
        
        # Check if FAISS index exists
        if not os.path.exists("faiss_index"):
            return jsonify({"error": "No document uploaded. Please upload a document first."}), 400
            
        print("Generating flashcards...")
        response = query_vector_store("""Extract key concepts from the uploaded document and generate flashcards in the following format:
Flashcard 1:
Question: [Concise and clear question]
Answer: [Brief, precise answer with relevant details]

Flashcard 2:
Question: [Concise and clear question]
Answer: [Brief, precise answer with relevant details]

Ensure that:
- Questions are clear and to the point.
- Answers are informative yet concise.
- Important technical terms are included.
- Each flashcard follows a structured and consistent format.
- Generate at least 5 flashcards covering the main topics.""")
        print("=== Flashcards completed successfully ===")
        return jsonify({"flashcards": response})
        
    except Exception as e:
        print(f"ERROR generating flashcards: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to generate flashcards", "details": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Simple chat endpoint for general questions."""
    global last_request_time
    
    # Rate limiting
    current_time = time.time()
    if current_time - last_request_time < MIN_REQUEST_INTERVAL:
        return jsonify({"error": "Please wait before making another request"}), 429
    last_request_time = current_time
    
    try:
        print("=== Chat Request ===")
        data = request.json
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"User question: {user_message}")
        
        # Create a simple prompt for general study assistance
        prompt_template = """
        You are StudyHelper, an AI assistant designed to help students with their studies. 
        You can help with:
        - Explaining concepts and topics
        - Solving math and science problems
        - Providing study tips and strategies
        - Answering academic questions
        - Helping with homework and assignments
        
        Please provide a clear, helpful, and educational response to the following question:
        
        Question: {question}
        
        Answer:
        """
        
        # Initialize the model with the correct model name
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        # Format the prompt
        formatted_prompt = prompt_template.format(question=user_message)
        
        # Get response from AI
        print("Sending to Gemini...")
        response = model.invoke(formatted_prompt)
        ai_response = response.content
        
        print(f"AI Response: {ai_response[:100]}...")
        print("=== Chat completed successfully ===")
        
        return jsonify({"response": ai_response})
        
    except Exception as e:
        print(f"ERROR in chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to process chat message", "details": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask AI Server...")
    print(f"Server will run on http://127.0.0.1:5000")
    print(f"Google API Key configured: {'Yes' if os.environ.get('GOOGLE_API_KEY') else 'No'}")
    app.run(debug=True, host='127.0.0.1', port=5000)
