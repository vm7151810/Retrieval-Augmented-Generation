# Importing necessary modules
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader as dl
from langchain.text_splitter import RecursiveCharacterTextSplitter as rcts
from langchain_community.vectorstores import Chroma
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA as qa
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

openai_key = "openai_key" # For accessing OpenAI services

# Function to load documents from a specified directory
def load_docs(dir):
    loader = dl(dir, show_progress=True) # Loading documents with progress bar
    docs = loader.load()
    return docs

# Function to split documents into smaller chunks for better processing
def split_text(docs):
    text_splitter = rcts(
        chunk_size=1000, # Maximum chunk size
        chunk_overlap=200, # Overlap between chunks to preserve context
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

# Function to generate embeddings from text chunks using OpenAI 
def get_embeddings(chunks):
    embedder = OpenAIEmbeddings(openai_api_key=openai_key)
    vector_db = Chroma.from_documents(documents=chunks, embedding=embedder) # Store embeddings in Chroma
    return vector_db

# Load the resumes from the specified directory
dir = "directory to the list of resumes"
docs = load_docs(dir)
print(len(docs))

chunks = split_text(docs)
print("Number of chunks created: ", len(chunks))

metadatas = [{"source": f"{i}-pl"} for i in range(len(chunks))] # Adding metadata for each chunk

vector_db = get_embeddings(chunks) # Get the vector embeddings for the chunks

# Retriever to find relevant chunks of text based on a query
retriever = vector_db.as_retriever(search_type= "similarity", search_kwargs={"k": 13}) 
#retrieved_docs = retriever.invoke("What is the educational background of name_of_candidate?")
#print(retrieved_docs[0].page_content)

# Prompt to guide the language model to extract specific information from the resumes
PROMPT = """
You have context from 10 resumes:
{context}

Your task is to extract and summarize key information from each of the 10 resumes. Specifically, you need to perform the following tasks for each resume:
1. Name Extraction: Identify the full name of the candidate from each resume.
2. Education Summary: Summarize the educational background in 1-2 sentences, mentioning key degrees, universities, and any notable achievements.
3. Experience Summary: Summarize the work experience by listing the companies, roles, and key responsibilities or achievements in 3-4 sentences.
4. Skills Extraction: Extract and summarize key skills based on the candidate's projects, work experience, and explicitly listed skills in 1-2 sentences.
5. Project Summary: Summarize each major project mentioned in the resume in 2-3 sentences, focusing on the objective, technology stack, and outcomes.

Return the output in the form of a dictionary with the following keys: `name`, `education`, `experience`, `skills`, `projects`. Ensure that there are a total of 10 candidates in the output.
"""

# Create a prompt template based on the provided prompt
QA_CHAIN_PROMPT = PromptTemplate.from_template(PROMPT) 

# Another prompt template for combining documents with their source information
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

# Creating a language model chain to process the data
llm_chain = LLMChain(
    llm=OpenAI(temperature=0), # OpenAI model with zero temperature for deterministic responses
    prompt=QA_CHAIN_PROMPT,
    callbacks=None, 
    verbose=True
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context", # Assigning the context variable
    document_prompt=document_prompt,
    verbose=True,
)

# Setting up the retrieval-based question-answering chain
qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    verbose=True,
    return_source_documents=True,
)  

# Perform the summary task using the QA chain
summary = qa(PROMPT)["result"]
print(summary)

job_description = "MS or PhD in computer science or a related technical field,5+ years of industry work experience. Good sense of product with a focus on shipping user-facing data-driven features, Expertise in Python and Python based ML/DL and Data Science frameworks. \
Excellent coding, analysis, and problem-solving skills. Proven knowledge of data structure and algorithms. \
Familiarity in relevant machine learning frameworks and packages such as Tensorflow, PyTorch and HuggingFace\
Experience working with Product Management and decomposing feature requirements into technical work items to ship products\
Experience with generative AI, knowledge of ML Ops and ML services is a plus. This includes Pinecone, LangChain, Weights and Biases etc. \
Familiarity with deployment technologies such as Docker, Kubernetes and Triton are a plus\
Strong communication and collaboration skills"

query = """
Based on the job description and the summary provided below, provide me the following:

1. Top Candidates Selection: Identify the top 3 candidates who best match the role of a Machine Learning Engineer.
2. Ranking and Summary: Rank the top 3 candidates in descending order of preference, and provide a brief summary of why each candidate is suitable for the role.

Provide the result in a structured format with the names of the top 3 candadidates suitable for the job.
"""

# Get the final answer based on the query, job description, and summary
Answer = qa(query + "\n" +job_description + "\n" + summary)["result"]
print(Answer)