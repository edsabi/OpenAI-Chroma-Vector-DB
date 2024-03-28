

with open('file.txt', 'r') as f:
    openai_key= f.read().strip()

import os

os.environ["OPENAI_API_KEY"] = openai_key

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader


loader = DirectoryLoader('./journals/', glob="./*.txt", loader_cls=TextLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

texts = text_splitter.split_documents(documents)


#print(len(texts))

persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts,embedding=embedding,persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)



