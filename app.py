from flask import Flask, request, jsonify
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
os.getenv("OPEN_API_KEY")

def get_conversational_chain():
    prompt_template = """
    Answer the question based on the context. If the answer isn't in the context, say "Answer not available in context". Be concise.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index5", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=2)  # Reduced from 5 to 2

    chain = get_conversational_chain()

    # Implement exponential backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            return response["output_text"]
        except Exception as e:
            if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        user_question = request.json['question']
        response = user_input(user_question)
        return jsonify({'response': response})
    except Exception as e:
        return josonify({'response':"Can You please rephrase you question"})
if __name__ == '__main__':
    app.run()

