import streamlit as st
from PyPDF2 import PdfReader
import hashlib
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# ✅ TEM QUE SER A PRIMEIRA COISA DO SCRIPT
st.set_page_config(
    page_title="📑 Analisador de Regulamentos Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentAnalyzer:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings()
        self.chat_model = ChatOpenAI(
            model_name="gpt-4",  # ou "gpt-3.5-turbo"
            temperature=0.3,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self.qa_chain = load_qa_chain(self.chat_model, chain_type="stuff")

        # Iniciar estado da sessão
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = None
        if "doc_hash" not in st.session_state:
            st.session_state.doc_hash = None
        if "history" not in st.session_state:
            st.session_state.history = []
        if "page_count" not in st.session_state:
            st.session_state.page_count = 0
        if "chunk_count" not in st.session_state:
            st.session_state.chunk_count = 0
        if "current_file" not in st.session_state:
            st.session_state.current_file = None
        if "current_file_hash" not in st.session_state:
            st.session_state.current_file_hash = None
        if "question_text" not in st.session_state:
            st.session_state.question_text = ""

    def _read_pdf(self, path: str) -> str:
        """Lê o conteúdo de um PDF e retorna como string"""
        pdf = PdfReader(path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        st.session_state.page_count = len(pdf.pages)
        return text

    def _split_text(self, text: str):
        """Divide o texto em partes menores (chunks)"""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(text)
        st.session_state.chunk_count = len(chunks)
        return chunks

    def _generate_embeddings(self, chunks: list):
        """Cria embeddings e armazena no vectorstore FAISS"""
        docs = [Document(page_content=chunk) for chunk in chunks]
        return FAISS.from_documents(docs, self.embedding_model)

    def _safe_rerun(self):
        """Força atualização da interface (gambiarra segura)"""
        st.experimental_rerun()

    def process_pdf(self, path: str):
        """Processa o PDF: leitura, chunk, embeddings"""
        with st.spinner("🔍 Processando documento..."):
            text = self._read_pdf(path)
            chunks = self._split_text(text)
            vectorstore = self._generate_embeddings(chunks)
            doc_hash = hashlib.sha256(open(path, 'rb').read()).hexdigest()
            return vectorstore, doc_hash, st.session_state.page_count, st.session_state.chunk_count

    def ask_question(self, question: str):
        """Executa a pergunta com base no documento carregado"""
        if not st.session_state.vectorstore:
            st.warning("Nenhum documento carregado ainda.")
            return

        docs = st.session_state.vectorstore.similarity_search(question)
        response = self.qa_chain.run(input_documents=docs, question=question)
        return response

    def _render_response(self, response: str):
        """Exibe a resposta formatada"""
        st.markdown("### 📬 Resposta")
        st.markdown(f"> {response}")

    def run(self):
        """Método principal para executar a aplicação"""
        try:
            with st.container():
                st.title("📑 Analisador de Leis e Regulamentos")
                st.markdown("""
                Carregue um arquivo PDF contendo leis, regulamentos ou qualquer outro documento oficial.
                Após o carregamento, você poderá fazer perguntas específicas sobre o conteúdo.
                """)

                uploaded_file = st.file_uploader("📤 Enviar documento PDF", type=["pdf"])

                if uploaded_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    current_file_hash = hashlib.sha256(open(tmp_path, 'rb').read()).hexdigest()

                    if current_file_hash != st.session_state.current_file_hash:
                        st.session_state.vectorstore, st.session_state.doc_hash, st.session_state.page_count, st.session_state.chunk_count = self.process_pdf(tmp_path)
                        st.session_state.current_file = uploaded_file.name
                        st.session_state.current_file_hash = current_file_hash
                        self._safe_rerun()

                if st.session_state.vectorstore:
                    st.success(f"📄 Documento carregado: **{st.session_state.current_file}**")
                    st.info(f"📄 Total de páginas: {st.session_state.page_count} | 🔍 Chunks: {st.session_state.chunk_count}")

                    st.session_state.question_text = st.text_input("❓ Faça sua pergunta sobre o documento:")

                    if st.session_state.question_text:
                        response = self.ask_question(st.session_state.question_text)
                        if response:
                            self._render_response(response)
                            st.session_state.history.append({
                                "pergunta": st.session_state.question_text,
                                "resposta": response
                            })

        except Exception as e:
            st.error(f"Erro na execução: {str(e)}")

# Executar o app
if __name__ == "__main__":
    app = DocumentAnalyzer()
    app.run()
