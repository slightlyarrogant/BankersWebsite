# response.py
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
# from main import vectorstore  # Import vectorstore from main.py

def get_answer(query, vectorstore):
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    When the question refers to McBride please treat it as a question referring to a Group.
    Please note that the Financial Year for McBride starts on 1st July and ends 30th June. 
    For example when we ask for financial data for 2022 we mean period starting from 1st July 2021 and lasting till 30th June 2022.
    Please provide references (name of the document and page) on the bottom of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model="llama2:13b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": query})
    return result
