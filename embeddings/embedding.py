import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
import pandas as pd

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Modificado: definir o caminho absoluto para 'carros.csv'
# pdf_path = "DATA-QUERY-AI/embeddings/ocorrencia.csv"
pdf_path = "embeddings\ocorrencia.csv"

loader = CSVLoader(pdf_path)
docs = loader.load()
print("docs: ", docs)

# # Se 'docs' for um DataFrame, converta-o em uma lista de strings
# if isinstance(docs, pd.DataFrame):
#     docs = docs.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()

# print("docs: ", docs)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )
# chunks = text_splitter.split_documents(
#     documents=docs,
# )

# persist_directory = 'db'

# embedding = OpenAIEmbeddings()
# vector_store = Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding,
#     persist_directory=persist_directory,
#     collection_name='ocorrencia',
# )