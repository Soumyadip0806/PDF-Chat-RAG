import streamlit as st
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import shutil
import base64
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
#import torch
#from transformers import AutoTokenizer , AutoModelForSeq2SeqLM, pipeline
#import torch
#from langchain.llms import HuggingFacePipeline
#from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#from langchain.chains import RetrievalQA



#embedding_model_local_path = "all-MiniLM-L6-v2"
#language_model_local_path = "LaMini-T5-738M"



# Retrive the PDF text
def get_pdf_text(pdf_documents):
    text = ""
    for pdf in pdf_documents:
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    return text if text.strip() else None


# Text to chunk Conversion
def get_text_chunk(text):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=5000, 
        chunk_overlap=1000,
        length_function=len)
    chunks = text_spliter.split_text(text)
    return chunks


# Chunk to Embedding and then store to vector database
def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    #embeddings= HuggingFaceBgeEmbeddings(model_name=embedding_model_local_path)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("database")
    return vectorstore



def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.5)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def get_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("database", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response


def clear_database():
    database_folder = "database"
    pdf_folder = "documents" 

    def clear_folder(folder):
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    st.error(f"Failed to delete {file_path}. Reason: {e}")
            return True
        return False

    db_cleared = clear_folder(database_folder)
    pdfs_cleared = clear_folder(pdf_folder)

    return db_cleared and pdfs_cleared



def displayPDF(folder_path):
    pdf_displays = "" 
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)  # Use full file path
        try: 
            with open(file_path, "rb") as f:  # Use file_path instead of filename
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="90%" height="400" type="application/pdf"></iframe>'
                pdf_displays += pdf_display + "<br><br>"
        except Exception as e:
            st.error(f"Failed to load PDF document '{filename}'. Error: {e}")
    
    return pdf_displays
    



'''

# This part is for local LLM models

language_model_local_path = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(language_model_local_path)
model = AutoModelForSeq2SeqLM.from_pretrained(
    language_model_local_path,
    offload_folder="./offload",
    device_map = "auto",
    torch_dtype = torch.float32,
    low_cpu_mem_usage=True,
    num_labels=3  # Specify a folder to offload weights to disk

    )

#model.to("cuda" if torch.cuda.is_available() else "cpu")



@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

#embedding_model_local_path = "all-MiniLM-L6-v2"
@st.cache_resource
def QA_llm():
    llm = llm_pipeline()
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    #embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_local_path)
    db = FAISS.load_local("faiss_index", embeddings)
    retriver = db.similarity_search()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriver = retriver,
        return_only_outputs = True
        #return_source_documents=True
    )
    return qa


def generate_answer(question):
    qa = QA_llm()
    generated_answer = qa(question)
    answer = generated_answer['result']
    return answer, generate_answer

'''

