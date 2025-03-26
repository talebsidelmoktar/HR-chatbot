# HR Policies Chatbot using FAISS, LangChain, and Streamlit
## Overview
This project allows a company to upload its HR policies in HTML format, convert them into vector embeddings using FAISS, and run a Streamlit-based chatbot to answer employee questions about company policies.

## Features
- Load HR policy documents from a directory.

- Split documents into smaller chunks for better retrieval.

- Generate embeddings using sentence-transformers.

- Store and retrieve documents using FAISS.

- Run a chatbot in Streamlit to answer employee queries.
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/talebsidelmoktar/HR-chatbot.git
   cd HR-chatbot
2. install the required depencies 
   ```bash
   pip install langchain langchain-community faiss-cpu sentence-transformers streamlit.

3. Query the FAISS Index
    ```bash
    python hrpc-FAISS-upload.py
The script will:

- Load the FAISS index.

- Perform similarity search on the indexed documents.

- Return the most relevant HR policy sections

4. Running the Chatbot in Streamlit
```bash
streamlit run hrpc-query.py

