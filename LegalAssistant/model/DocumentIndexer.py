import os
import faiss
import fitz  # PyMuPDF
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class PdfDocumentIndexer:
    def __init__(self, pdf_folder, index_folder="index_data", model_name='all-MiniLM-L6-v2'):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # self.pdf_folder = Path(pdf_folder)
        # self.index_folder = Path(index_folder)
        # self.index_file = self.index_folder / "faiss_index.index"
        # self.metadata_file = self.index_folder / "metadata.pkl"

        self.pdf_folder = Path(os.path.join(BASE_DIR, pdf_folder))
        self.index_folder = Path(os.path.join(BASE_DIR, index_folder))
        self.index_file = Path(os.path.join(self.index_folder, "faiss_index.index"))
        self.metadata_file = Path(os.path.join(self.index_folder, "metadata.pkl"))


        self.embed_model = SentenceTransformer(model_name)
        self.documents = []
        self.filenames = []
        self.index = None

        self.index_folder.mkdir(exist_ok=True)

        # if we have saved index and metadata, load them
        if self.index_file.exists() and self.metadata_file.exists():
            self._load_index()
        # else create a new index
        else:
            self._create_index()

    def _extract_text_from_pdfs(self):
        texts = []
        filenames = []

        for pdf_file in self.pdf_folder.glob("*.pdf"):
            try:
                doc = fitz.open(pdf_file)
                text = ""
                for i, page in enumerate(doc):
                    page_text = page.get_text()
                    text += page_text
                doc.close()

                if len(text.strip()) > 0:
                    texts.append(text)
                    filenames.append(pdf_file.name)
                else:
                    print(f"No extractable text found in {pdf_file.name}")

            except Exception as e:
                print(f"Failed to process {pdf_file.name}: {e}")
        return texts, filenames

    def _chunk_text(self, text, max_length=5000):
        # sentences = text.split('. ')
        # chunks, chunk = [], ""
        # for sentence in sentences:
        #     if len(chunk) + len(sentence) < max_length:
        #         chunk += sentence + ". "
        #     else:
        #         chunks.append(chunk.strip())
        #         chunk = sentence + ". "
        # if chunk:
        #     chunks.append(chunk.strip())
        # return chunks
        paragraphs = text.split('\n\n')  # Two newlines usually denote a paragraph
        chunks, chunk = [], ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(chunk) + len(paragraph) < max_length:
                chunk += paragraph + "\n\n"
            else:
                chunks.append(chunk.strip())
                chunk = paragraph + "\n\n"

        if chunk:
            chunks.append(chunk.strip())

        return chunks

    def _create_index(self):

        raw_texts, filenames = self._extract_text_from_pdfs()
        for text, fname in zip(raw_texts, filenames):
            chunks = self._chunk_text(text)
            if chunks:
                self.documents.extend(chunks)
                self.filenames.extend([fname] * len(chunks))
            else:
                print(f"Skipping {fname} — no chunks created")

            #self.documents.extend(chunks)
            #self.filenames.extend([fname] * len(chunks))

        # Before embedding, confirm we have documents
        #print(f"📄 Total chunks to embed: {len(self.documents)}")
        if not self.documents:
            raise ValueError("No documents were chunked. Check your PDFs and chunking logic.")
        
        embeddings = self.embed_model.encode(self.documents, convert_to_numpy=True)

        if embeddings.shape[0] == 0:
            raise ValueError("Embedding failed. No vectors returned.")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self._save_index()

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, 'wb') as f:
            pickle.dump({"documents": self.documents, "filenames": self.filenames}, f)

    def _load_index(self):
        #print("Loading FAISS index from disk...")
        self.index = faiss.read_index(str(self.index_file))
        with open(self.metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.filenames = data["filenames"]

    def search(self, query, top_k=3):
        query_vec = self.embed_model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(np.array(query_vec), top_k)
        results = [(self.documents[i], self.filenames[i], D[0][rank]) for rank, i in enumerate(I[0])]
        return results
