import streamlit as st
import pandas as pd
import uuid
import re
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# 🔹 Configuração do modelo OpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 🔹 Garante que cada sessão tenha um ID único
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 🔹 Simulação: os dados chegam como um DataFrame Pandas (com nulos)
def receber_dados():
    return pd.DataFrame({
        "mes": ["Janeiro", "Fevereiro", None, "Abril", "Maio"],
        "vendas": [1000, None, 1200, 1800, 1700]
    })

# 🔹 Armazena os dados no session_state
if "df" not in st.session_state:
    st.session_state.df = receber_dados()

df = st.session_state.df  # Carrega os dados da sessão

# 🔹 Esquema detalhado passado como dicionário
schema_info = {
    "mes": {"nome_logico": "Mês da Venda", "tipo": "string", "descricao": "Nome do mês referente às vendas"},
    "vendas": {"nome_logico": "Valor das Vendas", "tipo": "float", "descricao": "Total de vendas realizadas no mês"}
}

# 🔹 Tratamento de valores nulos
for coluna in df.columns:
    if df[coluna].dtype == "object":  # Se for string
        df[coluna].fillna("Desconhecido", inplace=True)
    else:  # Se for numérico
        df[coluna].fillna(0, inplace=True)

# 🔹 Formatando o esquema para o LLM
schema_str = "\n".join([
    f"- **{info['nome_logico']}** (`{col}`): {info['descricao']} (Tipo: {info['tipo']})"
    for col, info in schema_info.items()
])

# 🔹 Inicializa o histórico no session_state
if "historico" not in st.session_state:
    st.session_state.historico = []

# 🔹 Função para validar a query gerada
def validar_query(query):
    """Verifica se a query contém comandos proibidos."""
    padrao_proibido = r"\b(DROP|DELETE|ALTER|UPDATE|INSERT|REPLACE|MERGE|TRUNCATE)\b"
    return not re.search(padrao_proibido, query, re.IGNORECASE)

prefixo = f"""
Você está atuando como um assistente de dados e deve gerar consultas Pandas para responder perguntas sobre um DataFrame.

📌 **Regras:**
1️⃣ O esquema do DataFrame é o seguinte:
{schema_str}

2️⃣ Algumas colunas podem conter **valores nulos**. Esses valores já foram preenchidos automaticamente:
   - Strings vazias foram substituídas por `"Desconhecido"`.
   - Valores numéricos nulos foram substituídos por `0`.

3️⃣ As perguntas podem ter **contexto baseado no histórico da conversa**. 
Sempre leve em consideração o que já foi perguntado e respondido anteriormente.

4️⃣ Gere **somente a query Pandas**, sem explicações adicionais. 

🚫 **Restrições:** A consulta **NÃO pode modificar os dados** (exemplo: `DROP`, `DELETE`, `ALTER`, etc.).
"""

sufixo = """
Agora gere uma consulta Pandas para a seguinte pergunta, considerando o histórico da conversa:
{pergunta}
"""

# prompt_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content=prefixo),
#     HumanMessage(content=sufixo)
# ])

# 🔹 Interface Streamlit
st.title("🔍 Chat de Dados com RAG e Pandas")
st.write(f"🆔 **ID da Sessão:** {st.session_state.session_id}")

pergunta = st.text_input("📌 Faça uma pergunta sobre os dados:")

if pergunta:
    # 🔹 Mantém apenas as últimas 10 interações no histórico
    if len(st.session_state.historico) > 10:
        st.session_state.historico = st.session_state.historico[-10:]

    # 🔹 Adiciona a pergunta ao histórico
    st.session_state.historico.append(HumanMessage(content=pergunta))

    # 🔹 Constrói o prompt com o histórico da conversa
    prompt_historico = "\n".join([msg.content for msg in st.session_state.historico])
    prompt_completo = f"{prefixo}\n\n🔹 **Histórico da Conversa:**\n{prompt_historico}\n\n{sufixo.format(pergunta=pergunta)}"

    # 🔹 O LLM gera a consulta Pandas
    resposta_llm = llm.invoke(prompt_completo)
    query_pandas = resposta_llm.content.strip()

    # 🔹 Valida a query antes de executar
    if not validar_query(query_pandas):
        st.error("❌ **Erro:** A consulta gerada contém comandos proibidos!")
    else:
        # 🔹 Executa a consulta no DataFrame
        try:
            resposta = eval(query_pandas, {"df": df})  # Executa apenas com df no escopo seguro
            st.success("✅ **Consulta Gerada:**")
            st.code(query_pandas, language="python")

            # 🔹 Exibe o resultado
            st.write("📊 **Resultado:**")
            st.dataframe(resposta)

            # 🔹 Enriquecimento da resposta com LLM
            explicacao_prompt = f"""
            Você gerou a seguinte consulta Pandas:
            ```python
            {query_pandas}
            ```

            O resultado foi:
            ```
            {resposta.to_string(index=False)}
            ```

            Agora, forneça uma análise interpretando os dados e explicando o que eles representam.
            """
            explicacao_llm = llm.invoke(explicacao_prompt)
            explicacao = explicacao_llm.content.strip()

            # 🔹 Exibe a análise formatada como Markdown
            st.subheader("📢 Explicação da Resposta:")
            st.markdown(f"> {explicacao}")

            # 🔹 Adiciona a resposta ao histórico
            st.session_state.historico.append(SystemMessage(content=query_pandas))
            st.session_state.historico.append(SystemMessage(content=explicacao))

        except Exception as e:
            st.error(f"❌ **Erro ao executar a consulta:** {str(e)}")
