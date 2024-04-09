from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

app = app = Flask(__name__, template_folder='templates')


# Load OpenAI API Key from file
with open('file.txt', 'r') as f:
    openai_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = openai_key

persist_directory = 'db'

embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff", retriever=retriever, return_source_documents=True)



def process_prompt(prompt):
    llm_response = qa_chain(prompt)
    return llm_response

@app.route('/policybot')
def index():
    return render_template('index.html')

@app.route('/response', methods=['POST'])
def get_response():
    prompt = request.json['prompt']
    print(prompt+'\n')

   # persist_directory, embedding = setup_vectordb()
    llm_response = process_prompt(prompt)
    sources=[]

    for source in llm_response["source_documents"]:
        sources.append(source.metadata['source'])

    source_string='<br />Sources:<br />'
    for item in list(dict.fromkeys(sources)):
        source_string = source_string+str(item)




    response_text = llm_response['result']
    return jsonify({'result': response_text+'<br />'+source_string+'<br />'})
    print(response_text)
    print(source_string)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50050)
