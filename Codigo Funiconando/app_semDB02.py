import streamlit as st

# ⚠️ set_page_config deve ser a primeira chamada do script!
st.set_page_config(
    page_title="📑 Analisador de Regulamentos Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importações após o set_page_config
import os
import io
import hashlib
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Classe de análise
class DocumentAnalyzer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def _split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_text(text)

    def _generate_embeddings(self, chunks):
        return FAISS.from_texts(chunks, self.embeddings)

    def answer_question(self, vectorstore, question):
        chain = load_qa_chain(self.llm, chain_type="stuff")
        docs = vectorstore.similarity_search(question)
        return chain.run(input_documents=docs, question=question)

# Inicializa estados
for key in ["vectorstore", "current_file_hash", "current_file", "page_count", "chunk_count"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "page_count" and key != "chunk_count" else 0

st.title("📑 Analisador de Regulamentos CMB")
st.markdown("Envie um arquivo PDF com o regulamento e comece a fazer perguntas sobre ele.")

uploaded_file = st.file_uploader("Envie um arquivo PDF", type="pdf")
analyzer = DocumentAnalyzer()

if uploaded_file:
    uploaded_bytes = uploaded_file.read()
    current_file_hash = hashlib.sha256(uploaded_bytes).hexdigest()

    if current_file_hash != st.session_state.current_file_hash:
        with st.spinner("Processando documento..."):
            pdf_reader = PdfReader(io.BytesIO(uploaded_bytes))
            text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
            chunks = analyzer._split_text(text)
            vectorstore = analyzer._generate_embeddings(chunks)

            # Atualiza sessão
            st.session_state.vectorstore = vectorstore
            st.session_state.current_file_hash = current_file_hash
            st.session_state.current_file = uploaded_file.name
            st.session_state.page_count = len(pdf_reader.pages)
            st.session_state.chunk_count = len(chunks)

if st.session_state.vectorstore:
    st.success(f"📄 Documento processado com sucesso: {st.session_state.current_file}")
    st.write(f"🔢 Total de páginas: {st.session_state.page_count}")
    st.write(f"📚 Total de trechos (chunks): {st.session_state.chunk_count}")

    st.markdown("### ❓ Faça uma pergunta sobre o regulamento:")
    question = st.text_input("Digite sua pergunta:")

    if question:
        with st.spinner("Buscando resposta..."):
            answer = analyzer.answer_question(st.session_state.vectorstore, question)
            st.markdown("### ✅ Resposta:")
            st.write(answer)
