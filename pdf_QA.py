# Import necessary libraries
import PyPDF2
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter as rcts
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

llm = Ollama(model="llama3") # Initialize the language model

# Function to load and extract text from a PDF file
def load_pdf():
    pdf = PyPDF2.PdfReader(r"path_to_pdf")
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text() # Extract text from each page of the PDF
    return pdf_text

# Function to split the text into smaller chunks for processing
def split_text(text):
    text_splitter = rcts(
        chunk_size=1000, # Maximum size of each chunk
        chunk_overlap=100, # Overlap between chunks to maintain context
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

# Function to generate embeddings for each chunk and store them in FAISS
def get_embeddings(chunks):
    embedder = HuggingFaceEmbeddings()
    vector_storage = FAISS.from_texts(texts = chunks,metadatas=metadatas , embedding = embedder) # Store embeddings with FAISS
    return vector_storage

# Load and split the PDF text into chunks
pdf = load_pdf()
chunks = split_text(pdf)
print("Number of chunks created: ", len(chunks))
for i in range(len(chunks)):
    print()
    print(f"CHUNK : {i+1}")
    print(chunks[i])
    if (i > 5):
        break

metadatas = [{"source": f"{i}-pl"} for i in range(len(chunks))] # Assign metadata to each chunk for reference

vector_embeddings = get_embeddings(chunks)

# Set up a retriever to find relevant chunks based on the input query
retriever = vector_embeddings.as_retriever(search_type="similarity", search_kwargs={"k": 3}) 
#input_text = "Ask a question related to the uploaded pdf"
#retrieved_docs = retriever.invoke(input_text)

prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) # Create a prompt template object

# Initialize the LLM chain with the language model and prompt
llm_chain = LLMChain(
                  llm=llm, 
                  prompt=QA_CHAIN_PROMPT, 
                  callbacks=None, 
                  verbose=True)

# Define the document prompt for combining multiple documents
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

# Create a chain to combine retrieved documents and use the language model to answer questions
combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None,
              )

# RetrievalQA chain with the document combination chain and retriever
qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True,
              )

# function to process user questions and provide answers using the QA chain
def respond(question,history):
    return qa(question)["result"]

# Set up Gradio UI with a chat interface
gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me question related to the pdf", container=False, scale=7),
    title="Upload and Chat",
    examples=["How do lobsters fight?"],
    cache_examples=True,
    retry_btn=None,

).launch(share = True)


