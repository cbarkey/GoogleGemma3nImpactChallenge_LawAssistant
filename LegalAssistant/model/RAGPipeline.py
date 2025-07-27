from .DocumentIndexer import PdfDocumentIndexer
import os

import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

def load_rag_chain(pdf_folder = "data_pdf"):
    # Step 1: Create or load index
    indexer = PdfDocumentIndexer(pdf_folder=pdf_folder)

    index_folder="index_data"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    index_folder = os.path.join(BASE_DIR, index_folder)
    index_file = os.path.join(index_folder, "faiss_index.index")
    metadata_file = os.path.join(index_folder, "metadata.pkl")


    # Step 2: Load FAISS index and metadata
    print("Loading FAISS index from disk...")
    index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
        documents = metadata["documents"]
        filenames = metadata["filenames"]

    # Step 3: Convert to LangChain Documents
    docs = [
        Document(page_content=text, metadata={"source": filenames[i]})
        for i, text in enumerate(documents)
    ]
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

    # Step 4: Rebuild FAISS vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    # Step 5: Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20}, search_type = "mmr")

    # Step 6: Start Ollama with gemma3n:e4b model
    print("Starting Ollama with gemma3n:e4b model...")
    llm = OllamaLLM(model="gemma3n:e4b")

    # Step 7: Prompt template
    prompt_template = """
    You are a helpful legal assistant. Use the following context to answer the user's question.
    If the answer is not directly stated, try to infer it from the information provided.
    Always cite the source document.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Step 8: Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain

def get_answer(query: str, qa_chain) -> dict:
    result = qa_chain.invoke(query)
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
    }