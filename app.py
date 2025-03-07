import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
# Removido: from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# Removido: from langchain.agents.agent_types import AgentType

# Novas importações para o retriever com Chroma
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

def initial_parameters() -> tuple:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return model, parser, client

model, parser, client = initial_parameters()

# Carregar arquivo CSV e tratar valores ausentes
df = pd.read_csv('data/ocorrencia.csv', delimiter=";", encoding='latin1', on_bad_lines='skip').fillna(value=0)
print(df.head())
print(df.shape)

# Construir vetor de similaridade usando Chroma em memória
# Converter cada linha do DataFrame em um documento
docs = [Document(page_content=str(row.to_dict())) for _, row in df.iterrows()]
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma.from_documents(docs, embeddings, collection_name="data-query")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Função para perguntar com base nos documentos recuperados
def ask_question(model, retriever, query):
    # Instruções e prompt de contexto
    PROMPT_PREFIX = """
    - Recupere os nomes das colunas e, em seguida, responda à pergunta com base nos dados.
    - Caso a pergunta esteja fora do contexto, responda: "Não posso responder este tipo de pergunta, pois foge do contexto passado."
    """
    PROMPT_SUFFIX = """
    - **Antes de fornecer a resposta final**, utilize pelo menos um método adicional;
    reflita sobre ambos e verifique se os resultados respondem à pergunta original com precisão.
    - Formate números com quatro ou mais dígitos utilizando vírgulas.
    - Se os métodos divergem, reflita e tente outra abordagem; se ainda houver incerteza, reconheça.
    - A resposta final deve estar estruturada em markdown e sempre em Português Brasileiro.
    """
    # Obter documentos relevantes
    relevant_docs = retriever.get_relevant_documents(query)
    docs_context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"{PROMPT_PREFIX}\nContexto:\n{docs_context}\nPergunta: {query}\n{PROMPT_SUFFIX}"
    print("Prompt =", prompt)
    # Envio do prompt ao modelo
    response = model([{"role": "system", "content": prompt}])
    return response

st.title("Database AI Agent with Langchain")
st.write("### Dataset Preview")
st.write(df.head())

# Entrada de pergunta pelo usuário
st.write('### Ask a question')
# Valor padrão se nenhum input for fornecido inicialmente
question = st.chat_input()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).markdown(message.get('content'))
    st.chat_message('human').markdown(question)
    st.session_state.messages.append({'role': 'human', 'content': question})
    with st.spinner('Buscando resposta...'):
        response = ask_question(
            model=model,
            retriever=retriever,
            query=question
        )
        # Tratamento assumindo que o modelo retorne resposta no formato esperado
        output = response["output"] if "output" in response else response
        st.chat_message('ai').markdown(output)
        st.session_state.messages.append({'role': 'ai', 'content': output})