import streamlit as st
import sqlite3
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from dotenv import load_dotenv

# Bibliotecas IA
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configurações
load_dotenv()
st.set_page_config(page_title="Lab Manager Pro", page_icon="⚡", layout="wide")

# ==============================================================================
# 1. BANCO DE DADOS
# ==============================================================================
@st.cache_resource
def conectar_banco():
    conn = sqlite3.connect("lab_inversores_v5.db", check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS motores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tag_nome TEXT,
        tensao_v REAL,
        corrente_a REAL,
        potencia_cv REAL,
        polos INTEGER,
        rpm INTEGER
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inversores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        marca TEXT,
        modelo TEXT,
        corrente_nominal_a REAL,
        tensao_alimentacao TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS parametros (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_inversor INTEGER,
        id_motor INTEGER,
        codigo TEXT,
        descricao TEXT,
        valor_ajuste TEXT,
        FOREIGN KEY(id_inversor) REFERENCES inversores(id),
        FOREIGN KEY(id_motor) REFERENCES motores(id)
    )
    """)
    conn.commit()
    return conn

# ==============================================================================
# 2. FUNÇÕES AUXILIARES (PDF & RAG)
# ==============================================================================
@st.cache_resource
def processar_manual(arquivo_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(arquivo_pdf.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    os.remove(tmp_path)
    return vectorstore

def gerar_pdf_tecnico(inv_data, mot_data, params, aviso_subdimensionamento=False):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Relatório de Configuração de Teste", ln=True, align='C')
    
    if aviso_subdimensionamento:
        pdf.set_text_color(255, 0, 0) # Vermelho
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "⚠️ ATENÇÃO: TESTE A VAZIO (Inversor Subdimensionado)", ln=True, align='C')
        pdf.set_text_color(0, 0, 0) # Preto volta ao normal
        
    pdf.ln(5)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, "EQUIPAMENTOS", 1, 1, 'L', fill=True)
    pdf.set_font("Arial", '', 10)
    
    texto_inversor = f"INVERSOR: {inv_data[1]} {inv_data[2]} | Max: {inv_data[3]}A"
    texto_motor = f"MOTOR: {mot_data[1]} | Placa: {mot_data[3]}A | {mot_data[4]}CV"
    
    pdf.cell(0, 8, texto_inversor, 1, 1)
    pdf.cell(0, 8, texto_motor, 1, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(40, 10, "Parâmetro", 1)
    pdf.cell(110, 10, "Descrição", 1)
    pdf.cell(40, 10, "Valor Ajuste", 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 10)
    for p in params:
        pdf.cell(40, 10, str(p[0]), 1)
        pdf.cell(110, 10, str(p[1])[:60], 1)
        pdf.cell(40, 10, str(p[2]), 1)
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

# ==============================================================================
# INTERFACE
# ==============================================================================

conn = conectar_banco()
cursor = conn.cursor()

st.title("🏭 Lab Manager v5: Controle de Bancada")

tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Hardware", "📝 Mapear", "📄 PDF & Chat", "🤖 IA Parametrizadora"])

# --- TAB 1: CADASTRO (Simplificado para o exemplo) ---
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Novo Motor")
        with st.form("fm"):
            tag = st.text_input("Tag")
            cv = st.number_input("CV")
            amps = st.number_input("Corrente Nominal (A)")
            if st.form_submit_button("Salvar Motor"):
                cursor.execute("INSERT INTO motores (tag_nome, corrente_a, potencia_cv, tensao_v, polos, rpm) VALUES (?,?,?,380,4,1750)", (tag, amps, cv))
                conn.commit()
                st.success("Motor OK")
    with c2:
        st.subheader("Novo Inversor")
        with st.form("fi"):
            mod = st.text_input("Modelo")
            imax = st.number_input("Corrente Máxima (A)")
            if st.form_submit_button("Salvar Inversor"):
                cursor.execute("INSERT INTO inversores (marca, modelo, corrente_nominal_a, tensao_alimentacao) VALUES ('WEG', ?, ?, '380V')", (mod, imax))
                conn.commit()
                st.success("Inversor OK")

# --- TAB 2 e 3 (Mantidos similares, foco na TAB 4) ---
with tab2:
    st.write("Use esta aba para mapear manualmente se necessário (Código omitido para brevidade).")

with tab3:
    st.write("Chat Geral com o Banco de Dados (Código omitido para brevidade).")

# --- TAB 4: O CERÉBRO DO NEGÓCIO ---
with tab4:
    st.header("🤖 IA: Leitor de Manual com Ajuste de Carga")
    
    uploaded_file = st.file_uploader("1. Suba o Manual (PDF)", type="pdf")
    
    if uploaded_file:
        # Carrega dados para seleção
        df_mot = pd.read_sql("SELECT * FROM motores", conn)
        df_inv = pd.read_sql("SELECT * FROM inversores", conn)
        
        if not df_mot.empty and not df_inv.empty:
            c_sel1, c_sel2 = st.columns(2)
            
            # Seleção do Hardware
            idx_inv = c_sel1.selectbox("Inversor da Bancada:", df_inv.index, format_func=lambda x: f"{df_inv.iloc[x]['modelo']} ({df_inv.iloc[x]['corrente_nominal_a']}A)")
            idx_mot = c_sel2.selectbox("Motor sob Teste:", df_mot.index, format_func=lambda x: f"{df_mot.iloc[x]['tag_nome']} ({df_mot.iloc[x]['corrente_a']}A)")
            
            # Dados Reais
            inv_real = df_inv.iloc[idx_inv]
            mot_real = df_mot.iloc[idx_mot]
            
            st.divider()
            
            # === A LÓGICA DE NO-LOAD / SUBDIMENSIONAMENTO ===
            st.subheader("2. Cenário do Teste")
            
            # Checkbox crucial
            teste_a_vazio = st.checkbox("⚠️ Teste A VAZIO (Sem Carga no Eixo)", value=True, help="Marque isso se estiver usando um inversor menor que o motor apenas para girar em vazio.")
            
            corrente_para_ajuste = mot_real['corrente_a']
            aviso_ativo = False
            
            # Lógica Python de Segurança
            if teste_a_vazio:
                if inv_real['corrente_nominal_a'] < mot_real['corrente_a']:
                    st.warning(f"Detector de Subdimensionamento: O motor pede {mot_real['corrente_a']}A, mas o inversor só entrega {inv_real['corrente_nominal_a']}A.")
                    st.info(f"💡 A IA usará {inv_real['corrente_nominal_a']}A como base para os parâmetros de proteção, pois estamos em teste a vazio.")
                    corrente_para_ajuste = inv_real['corrente_nominal_a'] # AQUI É O PULO DO GATO
                    aviso_ativo = True
                else:
                    st.success("Inversor suporta o motor tranquilamente.")
            
            if st.button("🔍 Gerar Parâmetros Inteligentes"):
                with st.spinner("Processando Manual e Regras de Engenharia..."):
                    # Configura Vector Store
                    vectorstore = processar_manual(uploaded_file)
                    retriever = vectorstore.as_retriever()
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    
                    # Prompt Dinâmico
                    prompt_engenharia = f"""
                    Você é um técnico de bancada experiente.
                    
                    DADOS DO CENÁRIO:
                    - Inversor Limite: {inv_real['corrente_nominal_a']} A
                    - Motor Placa: {mot_real['corrente_a']} A
                    - Modo de Teste: {'TESTE A VAZIO (NO-LOAD)' if teste_a_vazio else 'CARGA NORMAL'}
                    
                    SUA MISSÃO:
                    Encontre os códigos dos parâmetros no manual PDF fornecido e sugira os valores.
                    
                    REGRA DE OURO (CRÍTICA):
                    1. Se o Modo de Teste for 'TESTE A VAZIO' e a corrente do motor ({mot_real['corrente_a']}A) for maior que a do inversor ({inv_real['corrente_nominal_a']}A):
                       - Para parâmetros de PROTEÇÃO (Corrente Limite, Sobrecarga): Use o limite do INVERSOR ({inv_real['corrente_nominal_a']}A).
                       - Para parâmetros de MODELAGEM (Corrente Nominal do Motor): Use o valor da PLACA DO MOTOR ({mot_real['corrente_a']}A) ou o do Inversor, o que for mais seguro para não desarmar o inversor. Explique sua escolha na descrição.
                    2. Recomende Modo de Controle ESCALAR (V/F) se houver subdimensionamento, pois Vetorial pode falhar no auto-tune.
                    
                    Saída esperada (Tabela Markdown):
                    | Código | Parâmetro | Valor Sugerido | Observação Técnica |
                    |---|---|---|---|
                    """
                    
                    # RAG Chain
                    docs = retriever.invoke("tensão nominal corrente nominal limite corrente modo controle parameters")
                    contexto = "\n\n".join([d.page_content for d in docs])
                    
                    resposta = llm.invoke(f"Manual Contexto:\n{contexto}\n\n{prompt_engenharia}")
                    
                    st.markdown(resposta.content)
                    
                    # Botão Simulado de Salvar (Para não complicar o exemplo com parser de markdown agora)
                    st.success("Se os valores acima estiverem corretos, copie-os para a Aba 2 para salvar no banco.")