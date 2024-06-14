import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv


load_dotenv()
os.getenv("OPEN_API_KEY")



def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer
    Take your time but donot provide answers that are not available in the context. also give a brief description of where you found the answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """


    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(OpenAI(), chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    return response["output_text"]

def main():
    st.set_page_config("infivent")
    st.header("""Infivent""")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    user_question = st.text_input("Ask a Question")

    if user_question:
        response = user_input(user_question)
        st.session_state.conversation_history.append("User: " + user_question)
        st.session_state.conversation_history.append("Bot: " + response)
        st.text_area("Conversation History", "\n".join(st.session_state.conversation_history), height=200)


if __name__ == "__main__":
    main()