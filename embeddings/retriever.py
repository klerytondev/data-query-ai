import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model='gpt-4',
)

persist_directory = 'db'
embedding = OpenAIEmbeddings()

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
    collection_name='carros',
)
retriever = vector_store.as_retriever()

system_prompt = '''
Use o contexto para responder as perguntas.
- Recupere os nomes das colunas e, em seguida, responda à pergunta com base nos dados.
- Caso a pergunta esteja fora do contexto, responda: "Não posso responder este tipo de pergunta, pois foge do contexto passado."
- **Antes de fornecer a resposta final**, utilize pelo menos um método adicional;
reflita sobre ambos e verifique se os resultados respondem à pergunta original com precisão.
- Formate números com quatro ou mais dígitos utilizando vírgulas.
- Se os métodos divergem, reflita e tente outra abordagem; se ainda houver incerteza, reconheça.
- A resposta final deve estar estruturada em markdown e sempre em Português Brasileiro.
Contexto: {context}
'''
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}'),
    ]
)
question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
)
chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

query = 'Qual é o carro mais novo?'

response = chain.invoke(
    {'input': query},
)
print(response)
