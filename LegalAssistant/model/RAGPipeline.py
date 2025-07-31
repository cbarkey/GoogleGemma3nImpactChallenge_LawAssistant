from .DocumentIndexer import PdfDocumentIndexer
import os

import faiss
import pickle
import time
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

class RAGPipelineClass:
    def __init__(self, pdf_folder = "data_pdf"):
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
        self.vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        # Step 5: Create retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20}, search_type = "mmr")

        # Step 6: Start Ollama with gemma3n:e4b model
        print("Starting Ollama with gemma3n:e4b model...")
        self.llm = OllamaLLM(model="gemma3n:e4b")

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
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        # return qa_chain

    def verify_generated_answer(self, original_query: str, answer_text: str, embedding_model_name="all-MiniLM-L6-v2", top_k=3):
        """
        Extract factual claims from an LLM-generated answer, embed them, and verify against a FAISS index.
        """

        # Step 1: Extract simple factual claims (placeholder - improve with NLP or LLM later)
        claims = [sentence.strip() for sentence in answer_text.split('.') if len(sentence.strip()) > 20]

        if not claims:
            return {"claims": [], "verified": []}

        # Step 2: Batch embed claims
        embedder = SentenceTransformer(embedding_model_name)
        claim_embeddings = embedder.encode(claims, convert_to_numpy=True)

        # Step 3: Search each claim embedding in the vectorstore
        verified_claims = []
        verified_claims_str = ""
        vector_search_time = 0
        llm_time = 0
        for i, embedding in enumerate(claim_embeddings):
            vs_time_start = time.time()
            results = self.vectorstore.similarity_search_by_vector(embedding, k=top_k)
            vs_time_end = time.time()
            vector_search_time += vs_time_end - vs_time_start
            # Very simple check: see if the claim's text appears in any of the top K docs
            supporting_docs = [doc.page_content for doc in results]
            # match_found = any(claims[i].lower() in doc.lower() for doc in supporting_docs)

            prompt = f"""Is this claim supported by the supporting documents? Answer with only 'True' or 'False'.
                      If the text is not a claim about something but simply supplementary text, answer with 'True'.
                      \n\nClaim: {claims[i]},\n\n
                      supporting documents: {supporting_docs}, \n\n
                      original query: {original_query}"""
            llm_time_start = time.time()
            llm_response = self.llm.invoke(prompt)
            llm_time_end = time.time()
            llm_time += llm_time_end - llm_time_start
            match_found = False
            if "True" in llm_response:
                match_found = True
                verified_claims_str += claims[i]
            elif  "False" in llm_response:
                match_found = False
            else:
                match_found = False

            verified_claims.append({
                "claim": claims[i],
                "supported": match_found,
                "sources": [doc.metadata.get("source", "Unknown") for doc in results]
            })
        
        print(f"LLM took {llm_time}s")
        print(f"Vector Search took {vector_search_time}s")

        # Have the LLM rewrite the response only with the claims that could be verified
        rewrite_prompt = f"""Please rewrite your original response with only the claims that were verified.
        If the claims were not verified, they should not be in the new response. Only respond with the new response, 
        there is no need to have any introduction text. \n\n
        original response: {answer_text}. \n\n
        verified claims: {verified_claims_str} """
        rewritten_response = self.llm.invoke(rewrite_prompt)

        return {
            "claims": claims,
            "verified": verified_claims,
            "rewritten response": rewritten_response
        }

    def get_answer(self, query: str) -> dict:
        start_time = time.time()
        # result = self.qa_chain.invoke(query)
        result = self.llm.invoke(query)
        llm_invoke_time = time.time()
        check_result = self.verify_generated_answer(query,result)
        end_time = time.time()
        time_diff = end_time - start_time
        print(f"Original LLM response took {llm_invoke_time - start_time}s")
        print(f"Took {time_diff}s to generate an answer")
        print(result)
        print(check_result["rewritten response"])
        print(check_result)

        # return {
        #     "answer": result["result"],
        #     "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        # }
        return {
            "answer": result,
            "sources": []
        }