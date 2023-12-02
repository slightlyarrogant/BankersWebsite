'''
This is the main file for the Bankers Website.
It contains the Flask application and routes for handling user requests.
'''
from flask import Flask, render_template, request
import response
from langchain.document_loaders import OnlinePDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sys

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

linklist = (
    "https://mcbride.co.uk/media/f0ictiu4/mcbride-fy23-year-end-release.pdf",
    "https://mcbride.co.uk/media/esclyhc1/nom_mcbride-2023.pdf",
    # "https://mcbride.co.uk/media/2b2peq5f/mcbride_fy2122_announcement.pdf",
    # "https://mcbride.co.uk/media/hr2heruc/mcbride-notice-of-meeting.pdf",
)

all_splits = []  # Initialize an empty list to collect all text splits

for element in linklist:
    print(element)
    loader = OnlinePDFLoader(element)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(data)
    all_splits.extend(splits)

with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = response.get_answer(question, vectorstore)['result']
    return render_template('index.html', question=question, answer=answer)

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

if __name__ == '__main__':
    app.run(debug=True)