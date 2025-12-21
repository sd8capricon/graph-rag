from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ingestion.readers.markdown import MarkdownReader

load_dotenv()


class Pipeline:

    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", output_dimensionality=768
    )

    def __init__(self, documents: list[Document], lexical_threshold: float = 0.75):
        self.vector_store = InMemoryVectorStore(self.embedding)
        self.documents = documents
        self.lexical_threshold = lexical_threshold

    def run(self):
        self._build_vectorstore()
        self._build_lexical_graph()

    def _build_vectorstore(self):
        doc_ids = [doc.metadata["id"] for doc in self.documents]
        self.document_ids = self.vector_store.add_documents(self.documents, doc_ids)

    def _build_lexical_graph(self):
        documentid_embeddings = self._get_documents_embedding(self.document_ids)
        edges: list[tuple[str, str]] = self._get_lexical_edges(documentid_embeddings)

    def _get_documents_embedding(self, ids: list[str]) -> dict[str, list]:
        embeddings_map: dict[str, list] = {}
        for doc_id, data in self.vector_store.store.items():
            vector = data["vector"]
            embeddings_map[doc_id] = vector
        return embeddings_map

    def _get_lexical_edges(
        self, id_embedding_map: dict[str, list]
    ) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        for doc_id, embedding in id_embedding_map.items():
            similarity_search = (
                self.vector_store.similarity_search_with_score_by_vector(embedding, 5)
            )
            for similar_document, score in similarity_search:
                if score > self.lexical_threshold:
                    edges.append((doc_id, similar_document.metadata["id"]))
        return edges


if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "data" / "alain_prost.md"
    reader = MarkdownReader(path)
    documents = reader.get_documents()
    pipeline = Pipeline(documents)
    pipeline.run()
