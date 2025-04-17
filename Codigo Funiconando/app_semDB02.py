import os
import hashlib
import tempfile
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# =============================================
# CONFIGURAÇÃO DA PÁGINA (DEVE SER O PRIMEIRO COMANDO STREAMLIT)
# =============================================
st.set_page_config(
    page_title="📑 Analisador de Regulamentos Pro",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# SOLUÇÃO MELHORADA PARA ERROS DE DOM
# =============================================
st.markdown("""
<script>
// Patch definitivo para erros de DOM com tratamento mais robusto
(function() {
    // 1. Patch para removeChild com verificação adicional
    const originalRemoveChild = Node.prototype.removeChild;
    Node.prototype.removeChild = function(child) {
        try {
            if (!child || !this || !this.contains || !this.contains(child)) {
                console.debug('[DOM Fix] Prevented removeChild error - invalid nodes');
                return child;
            }
            return originalRemoveChild.apply(this, arguments);
        } catch (e) {
            console.debug('[DOM Fix] Error in removeChild:', e);
            return child;
        }
    };
    
    // 2. Patch para insertBefore com tratamento de erro
    const originalInsertBefore = Node.prototype.insertBefore;
    Node.prototype.insertBefore = function(newNode, refNode) {
        try {
            if (!newNode || !this || !this.contains) {
                console.debug('[DOM Fix] Prevented insertBefore error - invalid nodes');
                return newNode;
            }
            if (refNode && (!this.contains || !this.contains(refNode))) {
                console.debug('[DOM Fix] Prevented insertBefore error - refNode not found');
                return newNode;
            }
            return originalInsertBefore.apply(this, arguments);
        } catch (e) {
            console.debug('[DOM Fix] Error in insertBefore:', e);
            return newNode;
        }
    };
    
    // 3. Patch para appendChild
    const originalAppendChild = Node.prototype.appendChild;
    Node.prototype.appendChild = function(child) {
        try {
            if (!child || !this || !this.contains) {
                console.debug('[DOM Fix] Prevented appendChild error - invalid nodes');
                return child;
            }
            return originalAppendChild.apply(this, arguments);
        } catch (e) {
            console.debug('[DOM Fix] Error in appendChild:', e);
            return child;
        }
    };
    
    console.log('Todos os patches de DOM foram aplicados com sucesso');
})();
</script>
""", unsafe_allow_html=True)

# =============================================
# CLASSE PRINCIPAL DO ANALISADOR (COM MELHORIAS)
# =============================================
class DocumentAnalyzer:
    def __init__(self):
        self._init_session_state()
        self._setup_environment()
    
    def _init_session_state(self):
        # Inicialização mais segura do session_state
        defaults = {
            'vectorstore': None,
            'doc_hash': None,
            'page_count': 0,
            'chunk_count': 0,
            'last_question': "",
            'show_response': None,
            'history': [],
            'current_file': None,
            'question_text': "",
            'processing': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _setup_environment(self):
        try:
            load_dotenv()
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                st.error("🔑 Chave da API não encontrada. Verifique seu arquivo .env")
                st.stop()
            
            genai.configure(api_key=self.api_key)
        except Exception as e:
            st.error(f"❌ Falha na configuração: {str(e)}")
            st.stop()
    
    def _safe_rerun(self):
        """Recarrega a página com proteção contra erros"""
        try:
            time.sleep(0.5)  # Tempo adicional para evitar condições de corrida
            st.rerun()
        except Exception as e:
            print(f"Erro durante rerun: {e}")
    
    def _get_embeddings(self):
        """Obtém embeddings com tratamento de erro"""
        try:
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
        except Exception as e:
            st.error(f"Erro ao obter embeddings: {str(e)}")
            st.stop()
    
    def process_pdf(self, file_path: str):
        """Processa o PDF com tratamento robusto de erros"""
        try:
            with st.status("📄 Processando documento...", expanded=True) as status:
                st.session_state.processing = True
                
                try:
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_documents(pages)
                    
                    doc_hash = hashlib.sha256()
                    for page in pages:
                        doc_hash.update(page.page_content.encode())
                    doc_hash = doc_hash.hexdigest()
                    
                    vectorstore = FAISS.from_documents(chunks, self._get_embeddings())
                    
                    status.update(
                        label=f"✅ Documento processado! (ID: {doc_hash[:12]}...)",
                        state="complete",
                        expanded=False
                    )
                    
                    return vectorstore, doc_hash, len(pages), len(chunks)
                finally:
                    st.session_state.processing = False
                    
        except Exception as e:
            st.session_state.processing = False
            status.update(label="❌ Falha no processamento", state="error")
            st.error(f"Erro no processamento: {str(e)}")
            return None, None, 0, 0
    
    def ask_question(self, question: str):
        """Processa perguntas com tratamento de erros melhorado"""
        try:
            if not question or not st.session_state.vectorstore:
                return "Por favor, carregue um documento e faça uma pergunta válida."
            
            prompt_template = """
            Você é um especialista em análise de documentos regulatórios. 
            Responda em português (Brasil) com tom profissional.
            
            Contexto: {context}
            Pergunta: {question}
            
            Instruções:
            - Formate a resposta com Markdown
            - Destaque artigos/seções com `código`
            - Use **negrito** para pontos importantes
            - Se não souber, diga "Não encontrado no documento"
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro-latest",
                    temperature=0.3,
                    google_api_key=self.api_key
                ),
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            result = qa_chain({"query": question})
            return result.get('result', "Não foi possível obter uma resposta.")
        except Exception as e:
            st.error(f"Erro durante a análise: {str(e)}")
            return None
    
    def run(self):
        """Método principal para executar a aplicação com tratamento de erros"""
        try:
            # Interface principal
            st.title("📑 Analisador de Leis e Regulamentos")
            st.markdown("Analise documentos regulatórios com IA. Carregue um PDF e faça perguntas sobre o conteúdo.")
            
            # Upload de arquivo com verificação de estado
            uploaded_file = st.file_uploader(
                "📤 Carregar regulamento (PDF)",
                type="pdf",
                help="Envie um documento PDF para análise",
                disabled=st.session_state.get('processing', False)
            )
            
            if uploaded_file and not st.session_state.processing:
                current_file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
                
                if 'current_file_hash' not in st.session_state or st.session_state.current_file_hash != current_file_hash:
                    st.session_state.current_file_hash = current_file_hash
                    st.session_state.vectorstore = None
                    
                    if uploaded_file.size > 10_000_000:
                        st.warning("Arquivos acima de 10MB podem demorar mais para processar.")
                    
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_file_path = tmp_file.name
                        
                        vectorstore, doc_hash, page_count, chunk_count = self.process_pdf(tmp_file_path)
                        
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                        
                        if vectorstore:
                            st.session_state.update({
                                'vectorstore': vectorstore,
                                'doc_hash': doc_hash,
                                'page_count': page_count,
                                'chunk_count': chunk_count
                            })
                            
                            with st.expander("📊 Resumo do documento"):
                                col1, col2 = st.columns(2)
                                col1.metric("Páginas", page_count)
                                col2.metric("Trechos", chunk_count)
                                st.caption(f"ID do documento: {doc_hash[:24]}...")
                    except Exception as e:
                        st.error(f"Erro ao processar arquivo: {str(e)}")
            
            if st.session_state.vectorstore and not st.session_state.processing:
                st.markdown("---")
                
                with st.form(key='question_form'):
                    question = st.text_input(
                        "💡 Faça sua pergunta sobre o regulamento:",
                        value=st.session_state.question_text,
                        placeholder="Ex: Quais são os requisitos para aprovação?"
                    )
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        submit_button = st.form_submit_button("🔍 Analisar", disabled=st.session_state.processing)
                    with col2:
                        if st.form_submit_button("🔄 Nova Pergunta", disabled=st.session_state.processing):
                            st.session_state.last_question = ""
                            st.session_state.show_response = None
                            st.session_state.question_text = ""
                            self._safe_rerun()
                
                if submit_button and question and not st.session_state.processing:
                    with st.spinner("🤖 Analisando pergunta..."):
                        try:
                            answer = self.ask_question(question)
                            if answer:
                                st.session_state.last_question = question
                                st.session_state.show_response = answer
                                st.session_state.question_text = ""
                                
                                if 'history' not in st.session_state:
                                    st.session_state.history = []
                                st.session_state.history.append({
                                    "question": question,
                                    "answer": answer,
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                                if len(st.session_state.history) > 5:
                                    st.session_state.history = st.session_state.history[-5:]
                                
                                self._safe_rerun()
                        except Exception as e:
                            st.error(f"Erro ao processar pergunta: {str(e)}")
                
                if st.session_state.show_response:
                    st.markdown(f"""
                    <div style='
                        padding: 1.5rem; 
                        background: white; 
                        border-radius: 12px; 
                        border-left: 4px solid #0a5c0a; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                        margin-bottom: 1rem;
                    '>
                        <h3 style='color: #6C63FF; margin-top: 0;'>📝 Resposta</h3>
                        {st.session_state.show_response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("❌ Limpar Resposta", disabled=st.session_state.processing):
                        st.session_state.show_response = None
                        self._safe_rerun()
                    
                    st.subheader("🔍 Trechos de referência")
                    try:
                        docs = st.session_state.vectorstore.similarity_search(
                            st.session_state.last_question, 
                            k=3
                        )
                        
                        for i, doc in enumerate(docs):
                            with st.expander(f"Trecho {i+1} (Página {doc.metadata.get('page', 'N/A')}"):
                                st.write(doc.page_content)
                    except Exception as e:
                        st.error(f"Erro ao recuperar trechos de referência: {str(e)}")
            
            # Sidebar com tratamento de erro
            try:
                with st.sidebar:
                    st.markdown("""
                    <div style='
                        padding: 1rem; 
                        background: #0a5c0a; 
                        color: white; 
                        border-radius: 12px;
                    '>
                        <h3 style='color: white !important;'>ℹ️ Como usar</h3>
                        <ol style='padding-left: 1rem;'>
                            <li>Carregue um PDF regulatório</li>
                            <li>Espere o processamento</li>
                            <li>Faça perguntas sobre o conteúdo</li>
                        </ol>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("📚 Histórico (Últimas 5)")
                    
                    if st.button("🧹 Limpar Todo o Histórico", disabled=st.session_state.processing):
                        st.session_state.history = []
                        self._safe_rerun()
                    
                    if 'history' in st.session_state and st.session_state.history:
                        for i, item in enumerate(reversed(st.session_state.history)):
                            with st.container():
                                st.markdown(f"""
                                <div style='
                                    padding: 0.5rem;
                                    margin-bottom: 0.5rem;
                                    border-radius: 8px;
                                    background: #f8f9fa;
                                    transition: all 0.2s;
                                '>
                                    <div style='font-size: 0.8rem; color: #6c757d;'>{item['timestamp']}</div>
                                    <p><strong>Pergunta:</strong> {item['question'][:60]}{'...' if len(item['question']) > 60 else ''}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if st.button(f"Ver resposta {i+1}", disabled=st.session_state.processing):
                                    st.session_state.show_response = item['answer']
                                    st.session_state.last_question = item['question']
                                    self._safe_rerun()
                    else:
                        st.caption("Nenhuma pergunta no histórico")
            except Exception as e:
                st.sidebar.error(f"Erro na barra lateral: {str(e)}")
        
        except Exception as e:
            st.error(f"Erro crítico: {str(e)}")
            self._safe_rerun()

# =============================================
# INICIALIZAÇÃO DA APLICAÇÃO
# =============================================
if __name__ == "__main__":
    try:
        analyzer = DocumentAnalyzer()
        analyzer.run()
    except Exception as e:
        st.error(f"Falha ao iniciar a aplicação: {str(e)}")
