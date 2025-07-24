import DocumentIndexer

import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document


def main():
    # Step 1: Create or load index
    indexer = DocumentIndexer.PdfDocumentIndexer(pdf_folder="data_pdf")

    # Step 2: Load FAISS index and metadata
    print("Loading FAISS index from disk...")
    index = faiss.read_index("index_data/faiss_index.index")
    with open("index_data/metadata.pkl", "rb") as f:
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Step 6: Start Ollama with gemma3:4b model
    print("Starting Ollama with gemma3:4b model...")
    llm = OllamaLLM(model="gemma3:4b")

    # Step 7: Prompt template
    prompt_template = """
    You are a helpful legal assistant. Use the following context to answer the user's question.

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

    # Step 9: Ask questions
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = qa_chain.invoke(query)

        print("\n🔎 Answer:\n", result["result"])
        print("\n📄 Sources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source", "Unknown"))

if __name__ == "__main__":
    main()
