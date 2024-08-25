import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_chat import message
import time
import os
from ingest import get_pdf_text, get_text_chunk, get_vectorstore, get_user_input, clear_database, displayPDF






os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


st.set_page_config(
    page_title="Chat with PDF",
    page_icon=":brain:",
    layout="centered",
)


def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))






def main():
    st.title("Chat with your PDF")
    chat_container = st.container()


# Sidebar Section
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_documents=st.file_uploader("Upload your PDFs here and click on the 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if clear_database():
                dbDelete_success_message=st.success("All old data has been cleared.")
                time.sleep(3) 
                dbDelete_success_message.empty()
            else:
                st.warning("No database found to clear.")
            if not pdf_documents:  # Check if no PDF files are uploaded
                st.error("Please upload at least one PDF file before processing.")
            else:
                with st.spinner("Processing..."):
                    raw_text=get_pdf_text(pdf_documents)
                    if raw_text is None:
                        st.error(f"The file contains no text and will be skipped.")
                    else:

                    # Save the uploaded PDFs to the folder and read their content
                        for pdf in pdf_documents:
                            save_path = os.path.join("documents", pdf.name) 
                            with open(save_path, mode='wb') as f:
                                f.write(pdf.getbuffer())  

                        text_chunks=get_text_chunk(raw_text)
                        #st.write(text_chunks)
                        get_vectorstore(text_chunks)
                        embedding_success_message=st.success("Embeddings have been successfully stored. You can start asking...")
                        time.sleep(3)  # Display the message for 3 seconds
                        embedding_success_message.empty()
                        
                        # Display each PDF
                        folder_path = "documents"
                        pdf_html = displayPDF(folder_path)
                        st.markdown(pdf_html, unsafe_allow_html=True)

 

# Initialize session state variables if they don't exist
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
    if "send_input" not in st.session_state:
        st.session_state.send_input = False


    user_input = st.text_input("Type here", key="user_input")     # Text input for the user
    send_button = st.button("Send", key="send_button")

    # Handle send button or text input submission
    if send_button or st.session_state.send_input:
        if user_input.strip():
            response = get_user_input(user_input)
            llm_response = response.get("output_text", "No response text found.")
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(llm_response)
            st.session_state.send_input = False  # Reset flag

    # Display the conversation
    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == '__main__':
    main()
