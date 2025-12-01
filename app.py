import streamlit as st
import sqlite3
import pandas as pd
import tempfile
import os
from dotenv import load_dotenv

# Config
load_dotenv() # Carrega a key de um arquivo .env

# Site hosteado no Streamlit 
if "OPENAI_API_KEY" in st.secrets: # Verifica se a key está na aba Secrets do Streamlit
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] # Define a key para uso no sistema

from langchain_community.document_loaders import PyPDFLoader # Lê PDF
from langchain_community.vectorstores import FAISS # DB vetorial 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # Tradutor de texto em número (DB Vetorial) e chatgpt
from langchain_text_splitters import RecursiveCharacterTextSplitter # Separa textos grandes em partes

# Nome da página
st.set_page_config(page_title="Faraday Pilot", layout="wide")

# Banco de dados

@st.cache_resource # Conecta no banco uma única vez e guarda essa informação na memória
def conectar_banco():
    # Conecta ao banco
    conn = sqlite3.connect("lab_manager_v7.db", check_same_thread=False) # Cria o arquivo .db
    cursor = conn.cursor() # Lê e escreve no banco
    
    # Tabela motores
    # Envia um comando que, caso não haja a seguinte tabela, cria uma tabela
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

    # Tabela inversores
    # Envia um comando que, caso não haja a seguinte tabela, cria uma tabela
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inversores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        marca TEXT,
        modelo TEXT,
        corrente_nominal_a REAL,
        potencia_cv REAL, 
        tensao_alimentacao TEXT
    )
    """)
    
    # Tabela parametros
    # Envia um comando que, caso não haja a seguinte tabela, cria uma tabela
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
    
    # Banco para testes
    # Verifica se a tabela de inversores está vazia
    # Caso esteja, ele insere os seguintes dados para usos em testes
    cursor.execute("SELECT count(*) FROM inversores")
    if cursor.fetchone()[0] == 0:
        print("Banco vazio detectado. Inserindo dados de exemplo...")
        
        # Lista de Inversores
        lista_inversores = [
            # WEG 
            ('WEG', 'CFW500', 2.6, 1.0, '220V'),
            ('WEG', 'CFW500', 4.3, 1.5, '220V'),
            ('WEG', 'CFW11', 10.0, 5.0, '380V'),
            ('WEG', 'CFW11', 24.0, 15.0, '380V'),
            ('WEG', 'CFW11', 142.0, 100.0, '380V'), # Para teste de subdimensionamento
            
            # Danfoss 
            ('Danfoss', 'VLT Micro Drive FC 51', 2.2, 0.75, '220V'),
            ('Danfoss', 'VLT Micro Drive FC 51', 6.8, 2.0, '220V'),
            ('Danfoss', 'VLT AutomationDrive FC 302', 16.0, 10.0, '380V'),
            ('Danfoss', 'VLT AQUA Drive FC 202', 32.0, 20.0, '380V'),
            
            # ABB 
            ('ABB', 'ACS150', 4.7, 1.5, '220V'),
            ('ABB', 'ACS380', 5.6, 3.0, '380V'),
            ('ABB', 'ACS580', 17.0, 10.0, '380V'),
            ('ABB', 'ACS880 Industrial', 65.0, 40.0, '380V')
        ]
        
        cursor.executemany("""
            INSERT INTO inversores (marca, modelo, corrente_nominal_a, potencia_cv, tensao_alimentacao)
            VALUES (?, ?, ?, ?, ?)
        """, lista_inversores) # Insere os seguintes dados da lista inversores na tabela motores no db
        
        # Lista de motores
        lista_motores = [
            ('Motor Bancada A (Pequeno)', 220, 2.8, 1.0, 4, 1720),
            ('Motor Bancada B (Médio)', 380, 7.5, 5.0, 4, 1750),
            ('Motor Esteira (Grande)', 380, 22.5, 15.0, 4, 1760),
            ('Motor Compressor (Extra Grande)', 380, 138.0, 100.0, 2, 3550)
        ]
        
        cursor.executemany("""
            INSERT INTO motores (tag_nome, tensao_v, corrente_a, potencia_cv, polos, rpm)
            VALUES (?, ?, ?, ?, ?, ?)
        """, lista_motores) # Insere os seguintes dados da lista motores na tabela motores no db

        # Salva no banco
        conn.commit()
    
    # \re
    return conn

# RAG (Leitura dos PDFs)
@st.cache_resource # Para ler o pdf uma única vez e guardar essa informação na memória 
def processar_manual(arquivo_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file: # Cria um PDF temporário
        tmp_file.write(arquivo_pdf.read()) # Escreve o conteudo do upload (manual enviado) e no arquivo temporário
        tmp_path = tmp_file.name # Salva o caminho do arquivo para usarmos no PyPDFLoader

    loader = PyPDFLoader(tmp_path) # Lê o arquivo no tmp_path
    docs = loader.load() # Abre o arquivo, extrai todo o texto e transforma em paginas de strings 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300) # Chama a ferramenta e dimensiona o tamanho da chunk
    splits = text_splitter.split_documents(docs) # Transformas a paginas em chunks menores
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings()) # Traduz essas chunks de strings em vetores númericos e armazena no banco (FAISS)
    os.remove(tmp_path) # Exclui o arquivo temporario
    return vectorstore # Retorna o banco de vetores

# Interface
conn = conectar_banco() 
cursor = conn.cursor()

st.title("Faraday Pilot: Assistente de Parametrização")

# Abas
tab1, tab2, tab_ia = st.tabs(["Cadastro Hardware", "Estoque Cadastrado", "IA Parametrizadora"])

# Tabela 1 - Cadastro de motores e inversores
with tab1:
    c1, c2 = st.columns(2) # Cria duas colunas, uma para motores e outra para inversores
    
    # Aba motor
    with c1:
        st.subheader("Novo Motor")
        # Forms motor
        with st.form("fm_motor", clear_on_submit=True): # Evita com que a pagina atualize a cada interação e, ao salvar, apaga o que está inscrito em todos os campos
            tag = st.text_input("Tag / Nome", placeholder="Ex: Motor Bancada A") # Nome do motor
            # Cria duas colunas
            col_a, col_b = st.columns(2)
            cv = col_a.number_input("Potência (CV)", min_value=0.1, step=0.1) # Potência do motor
            volts = col_b.number_input("Tensão (V)", value=380) # Tensão do motor
            amps = col_a.number_input("Corrente (A)", min_value=0.1, step=0.1) # Corrente do motor
            rpm = col_b.number_input("Rotação (RPM)", value=1750) # RPM do motor
            polos = st.selectbox("Polos", [2, 4, 6, 8], index=1) # Polos do motor
            
            # Cria um botão para salvar o forms
            if st.form_submit_button("Salvar Motor"):
                #Executa a entrada de dados do motor na tabela motores
                cursor.execute("""
                    INSERT INTO motores (tag_nome, tensao_v, corrente_a, potencia_cv, polos, rpm) 
                    VALUES (?,?,?,?,?,?)""", (tag, volts, amps, cv, polos, rpm))
                # Salva os dados enviados na tabela
                conn.commit()
                st.success("Motor cadastrado com sucesso!")

    # Aba inversor
    with c2:
        st.subheader("Novo Inversor")
        # Forms inversor
        with st.form("fm_inversor", clear_on_submit=True): # Evita com que a pagina atualize a cada interação e, ao salvar, apaga o que está inscrito em todos os campos
            marca = st.text_input("Marca", placeholder="WEG") # Marca do inversor
            mod = st.text_input("Modelo", placeholder="CFW11") # Modelo do inversor
            # Cria duas colunas
            ci1, ci2 = st.columns(2) 
            imax = ci1.number_input("Corrente Nominal (A)", min_value=1.0, step=0.1) # Corrente do inversor
            pot_inv = ci2.number_input("Potência Máxima (CV)", min_value=0.1, step=0.1) # Potência do inversor
            tensao_in = st.selectbox("Alimentação", ["220V", "380V", "440V"], index=1) # Tensão do inversor
            
            # Cria um botão para salvar o forms
            if st.form_submit_button("Salvar Inversor"):
                # Exexcuta a entrada de dados do inversor na tabela inversores
                cursor.execute("""
                    INSERT INTO inversores (marca, modelo, corrente_nominal_a, potencia_cv, tensao_alimentacao) 
                    VALUES (?,?,?,?,?)""", (marca, mod, imax, pot_inv, tensao_in))
                # Salva os dados enviados na tabela
                conn.commit()
                st.success("Inversor cadastrado!")

# Tabela 2 - estoque
with tab2:
    st.header("Inventário de Equipamentos")
    st.markdown("Abaixo estão listados todos os equipamentos disponíveis para teste.")
    
    # Cria duas colunas
    col_mot, col_inv = st.columns(2)
    
    with col_mot:
        st.subheader("Motores")
        df_m = pd.read_sql("SELECT * FROM motores", conn) # Lê e e transforma o conteúdo em um dataframe
        # Mostra a tabela (dataframe) dos motores cadastrados
        if not df_m.empty:
            st.dataframe(df_m, use_container_width=True, hide_index=True)
        # Caso esteja vazio, exibe essa mensagem
        else:
            st.info("Nenhum motor cadastrado ainda.")

    with col_inv:
        st.subheader("Inversores")
        df_i = pd.read_sql("SELECT * FROM inversores", conn) # Lê e transforma o conteúdo em um dataframe
        # Mostra a tabela (dataframe) dos inversores cadastrados
        if not df_i.empty:
            st.dataframe(df_i, use_container_width=True, hide_index=True)
        # Caso esteja vazio, exibe essa mensagem
        else:
            st.info("Nenhum inversor cadastrado ainda.")
            
    st.divider()
    st.caption("Nota: Para atualizar as tabelas após um cadastro, basta clicar na aba novamente ou recarregar a página.")

# tabela 3 - Análise de manuai e cruzamento de dados
with tab_ia:
    st.header("Análise Inteligente de Manual")
    
    uploaded_file = st.file_uploader("Carregue o Manual (PDF)", type="pdf")
    
    df_mot = pd.read_sql("SELECT * FROM motores", conn)
    df_inv = pd.read_sql("SELECT * FROM inversores", conn)
    
    if uploaded_file and not df_mot.empty and not df_inv.empty:
        c_sel1, c_sel2 = st.columns(2)
        
        idx_inv = c_sel1.selectbox("Inversor:", df_inv.index, format_func=lambda x: f"{df_inv.iloc[x]['modelo']} ({df_inv.iloc[x]['corrente_nominal_a']}A | {df_inv.iloc[x]['potencia_cv']}CV)")
        idx_mot = c_sel2.selectbox("Motor:", df_mot.index, format_func=lambda x: f"{df_mot.iloc[x]['tag_nome']} ({df_mot.iloc[x]['corrente_a']}A | {df_mot.iloc[x]['potencia_cv']}CV)")
        
        inv_real = df_inv.iloc[idx_inv]
        mot_real = df_mot.iloc[idx_mot]
        
        st.divider()
        
        teste_a_vazio = st.checkbox("Modo Teste A VAZIO (Subdimensionamento Permitido)", value=True)
        
        # Logica do calculo
        alvo_corrente = mot_real['corrente_a']
        alvo_potencia = mot_real['potencia_cv']
        alvo_tensao = mot_real['tensao_v']
        alvo_rpm = mot_real['rpm']
        alvo_polos = mot_real['polos']
        
        aviso_sub = False
        
        if teste_a_vazio and (mot_real['corrente_a'] > inv_real['corrente_nominal_a']):
            aviso_sub = True
            alvo_corrente = inv_real['corrente_nominal_a']
            if mot_real['potencia_cv'] > inv_real['potencia_cv']:
                 alvo_potencia = inv_real['potencia_cv']
        
        if aviso_sub:
            st.warning(f"Subdimensionamento Detectado! Ajustando valores de segurança:")
            st.code(f"""
            Motor Real: {mot_real['corrente_a']}A | {mot_real['potencia_cv']}CV
            Inversor:   {inv_real['corrente_nominal_a']}A | {inv_real['potencia_cv']}CV
            
            >> VALORES QUE SERÃO USADOS NA PARAMETRIZAÇÃO:
            Corrente Alvo: {alvo_corrente} A (Limitada pelo Inversor)
            Potência Alvo: {alvo_potencia} CV (Limitada pelo Inversor)
            """)
        
        if st.button("Gerar Parâmetros"):
            with st.spinner("Lendo manual e cruzando dados..."):
                try:
                    vectorstore = processar_manual(uploaded_file)
                    retriever = vectorstore.as_retriever()
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    
                    prompt_sistema = f"""
                    Você é um especialista em parametrização de drives.
                    OBJETIVO: Encontrar os CÓDIGOS dos parâmetros no manual e preencher com os VALORES ALVO que eu calculei.
                    
                    DADOS TÉCNICOS OBRIGATÓRIOS (Use estes valores exatos):
                    1. Corrente Nominal do Motor: {alvo_corrente} A
                    2. Potência Nominal do Motor: {alvo_potencia} CV
                    3. Tensão Nominal do Motor: {alvo_tensao} V
                    4. Rotação Nominal (RPM): {alvo_rpm} RPM
                    5. Número de Polos: {alvo_polos}
                    
                    INSTRUÇÃO:
                    - Procure no texto fornecido qual é o código (Ex: P0401, P-02) para cada item.
                    - Se houver subdimensionamento, recomende modo de controle V/F (Escalar).
                    
                    SAÍDA (Tabela Markdown):
                    | Código | Descrição do Parâmetro | Valor Ajuste | Observação |
                    |---|---|---|---|
                    """
                    
                    docs = retriever.invoke(f"parâmetros nominais motor tensão corrente potência polos rpm {inv_real['modelo']}")
                    contexto = "\n".join([d.page_content for d in docs])
                    res = llm.invoke(f"Manual:\n{contexto}\n\n{prompt_sistema}")
                    
                    st.success("Sugestão de Parametrização Gerada:")
                    st.markdown(res.content)
                    
                    st.divider()
                    st.subheader("Exportar Dados")
                    
                    dados_exportacao = pd.DataFrame([
                        {"Parametro": "Corrente Nominal (Motor)", "Valor": f"{alvo_corrente} A", "Nota": "Ajustado p/ Inversor" if aviso_sub else "Nominal"},
                        {"Parametro": "Potência (Motor)", "Valor": f"{alvo_potencia} CV", "Nota": "Ajustado p/ Inversor" if aviso_sub else "Nominal"},
                        {"Parametro": "Tensão Nominal", "Valor": f"{alvo_tensao} V", "Nota": "Mantido"},
                        {"Parametro": "Rotação (RPM)", "Valor": f"{alvo_rpm} RPM", "Nota": "Mantido"},
                        {"Parametro": "Polos", "Valor": str(alvo_polos), "Nota": "Mantido"},
                        {"Parametro": "Modo de Controle", "Valor": "V/F (Escalar)" if aviso_sub else "Vetorial", "Nota": "Recomendação IA"}
                    ])
                    
                    csv = dados_exportacao.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Baixar Arquivo de Parâmetros (.csv)",
                        data=csv,
                        file_name=f"param_inv_{inv_real['modelo']}_mot_{mot_real['tag_nome']}.csv",
                        mime="text/csv",
                    )
                    
                except Exception as e:
                    st.error(f"Erro: {e}")