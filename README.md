<h1>Document and Resume Summarization with Q&A</h1>
<h3>This project leverages LangChain, OpenAI(and Ollama), and Chroma(and FAISS) to perform two main tasks:</h4>
<ul>
  <li>PDF-based Question & Answer System: Upload a PDF and ask questions about its content.</li>
  <li>Resume Summarization & Candidate Selection: Summarize resumes and select the top candidates for a Machine Learning Engineer role based on a job description.</li>
</ul>
<br>
<h2>Features:</h2>
<h3>1. PDF-based Question & Answer System</h3>
<ul>
  <li>Upload PDFs: Load any PDF document into the system.</li>
  <li>Text Extraction: Automatically extract and split text from the uploaded PDF into manageable chunks.</li>
  <li>Embedding & Retrieval: Use OpenAI or HuggingFace embeddings for text chunking and retrieval.</li>
  <li>Q&A: Ask questions related to the uploaded PDF, and the system retrieves the relevant context and provides an answer using a language model.</li>
  <li>Interactive Chat Interface: Provides a user-friendly Gradio-based chat interface for interacting with PDF content.</li>
</ul>

<h3>2. Resume Summarization & Candidate Selection</h3>
<ul>
  <li>Resume Loading: Load resumes from a specified directory.</li>
  <li>Text Chunking: Split resumes into smaller chunks for processing.</li>
  <li>Embedding: Generate embeddings using OpenAI's API for similarity search.</li>
  <li>Information Extraction: Automatically extract:</li>
    <ol>
      <li>Candidate name</li>
      <li>Educational background</li>
      <li>Work experience</li>
      <li>Key skills and project summaries</li>
    </ol>
  <li>Candidate Selection: Based on a job description, identify and rank the top 3 candidates who are the best match for a Machine Learning Engineer role.</li>
</ul>
<br>
<h2>Usage</h2>
<h3>1. Install dependencies</h3>
<li>Common Dependencies</li>
<pre><code>pip install PyPDF2
pip install langchain
</code></pre>
<li>For PDF-based Q&A Project</li>
<pre><code>pip install langchain_huggingface
pip install gradio
pip install faiss-cpu
</code></pre>
<li>For Resume Summarization Project</li>
<pre><code>pip install chromadb
pip install openai
</code></pre>
<h4>2. Update the directories to the pdf and resumes</h4>
<h4>3. Set your OpenAI API key in the openai_key variable</h4>
<br>
<h1>Ready to use!!</h1>
