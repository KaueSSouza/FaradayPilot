#Faraday Pilot: Assistente de Parametrização Industrial

> Um Agente Autônomo que combina RAG (Leitura de Manuais) e Text-to-SQL (Inventário) para automatizar testes de engenharia e garantir segurança operacional.

![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Stack](https://img.shields.io/badge/Tech-LangChain%20|%20Streamlit%20|%20OpenAI-orange)

#O Problema
Na rotina de testes de inversores de frequência, técnicos perdem tempo consultando manuais extensos (PDFs) e correm risco de configurar parâmetros errados ao testar motores de diferentes potências.

#A Solução
Desenvolvi uma aplicação que atua como um **Engenheiro Sênior Digital**:
1.  **Lê o Manual na hora:** O usuário sobe o PDF e a IA indexa o conteúdo (RAG).
2.  **Consulta o Estoque:** O sistema sabe quais motores temos na bancada (Banco SQL).
3.  **Aplica Regras de Engenharia:** Detecta automaticamente se o inversor é menor que o motor (teste a vazio) e força parâmetros de segurança.

#Arquitetura Técnica

O projeto utiliza uma abordagem híbrida de Agentes:
* **Frontend:** Streamlit (Interface de Chat e Gestão).
* **Cérebro (LLM):** GPT-4o-mini via LangChain.
* **Memória 1 (Vetorial):** FAISS para busca semântica nos manuais PDF.
* **Memória 2 (Relacional):** SQLite para dados estruturados de hardware.
* **Orquestração:** LangChain Agents com Tools personalizadas.

# Como Rodar

1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
