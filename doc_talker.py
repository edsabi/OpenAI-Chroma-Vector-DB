with open('file.txt', 'r') as f:
    openai_key= f.read().strip()

import os 

os.environ["OPENAI_API_KEY"] = openai_key

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)

retriever = vectordb.as_retriever()



qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff", retriever=retriever, return_source_documents=True)



def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


while True:
    prompt = input('prompt:')
    llm_response = qa_chain(prompt)
    process_llm_response(llm_response)
    llm_response




