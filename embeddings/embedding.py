import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings


os.environ["OPENAI_API_KEY"] = 'SUA CHAVE DE API'

pdf_path = 'carros.csv'
loader = CSVLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(
    documents=docs,
)

persist_directory = 'db'

embedding = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name='carros',
)
