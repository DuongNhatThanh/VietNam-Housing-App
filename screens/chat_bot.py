import streamlit as st
#Import library
import yaml
#load config.yml and parse into variables 
with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
_BARD_API_KEY = cfg["API_KEY"]["Bard"]
main_path = cfg["LOCAL_PATH"]["main_path"]
chat_context_length = cfg["CHAT"]["chat_context_length"]
model_name = cfg["EMBEDDINGS"]["HuggingFaceEmbeddings"]["model_name"]
model_kwargs = cfg["EMBEDDINGS"]["HuggingFaceEmbeddings"]["model_kwargs"]
chunk_size = cfg["CHUNK"]["chunk_size"]
chunk_overlap = cfg["CHUNK"]["chunk_overlap"]

from langchain.vectorstores import Chroma
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# Bard
from bardapi import Bard
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from streamlit_feedback import streamlit_feedback


#define Bard
class BardLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response = Bard(token=_BARD_API_KEY).get_answer(prompt)['content']
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}



def load_embeddings(): 
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    chroma_index = Chroma(persist_directory=main_path+"/vectorstore/chroma_db", embedding_function=embeddings)
    print("Successfully loading embeddings and indexing")
    return chroma_index



def ask_with_memory(vector_store, question, chat_history=[], document_description=""):

    llm=BardLLM()
    retriever = vector_store.as_retriever( # now the vs can return documents
    search_type='similarity', search_kwargs={'k': 3})
 
    general_system_template = f"""
    You are a professional consultant at a real estate consulting company, providing consulting services \
    to customers on real estate development strategies, real estate news and real estate law.\
    Your role is to communicate with customer, then interact with them about their concerns about real estates.\
    Once the customer has been provided their question,\
    then you obtain some documents about real estate laws or real estate news related to their question.\
    Then you will examine these documents .\
    You must provide the answer based on these documents which means\
    using only the heading and piece of context to answer the questions at the end.\
    If you don't know the answer just say that you don't know, don't try to make up an answer. \
    If the question is not in the field of real estate , just answer that you do not know. \
    You respond in a short, very conversational friendly style.\
    Answer only in Vietnamese\
    ----
    HEADING: ({document_description})
    CONTEXT: {{context}}
    ----
    """
    general_user_template = """Here is the next question, remember to only answer if you can from the provided context. 
    If the question is not relevant to real estate , just answer that you do not know, do not create your own answer.
    Only respond in Vietnamese.
     QUESTION:```{question}```"""

    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )


    crc = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs={'prompt': qa_prompt})
    result = crc({'question': question, 'chat_history': chat_history})
    return result


def clear_history():
    if "history" in st.session_state:
        st.session_state.history = []
        st.session_state.messages = [] 

# Define a function for submitting feedback
def _submit_feedback(user_response, emoji=None):
    st.toast(f"Feedback submitted: {user_response}", icon=emoji)
    return user_response.update({"some metadata": 123})


def format_chat_history(chat_history):
    formatted_history = ""
    for entry in chat_history:
        question, answer = entry
        # Added an extra '\n' for the blank line
        formatted_history += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_history

def run_chatbot():
    with st.sidebar.title("Sidebar"):
        if st.button("Clear History"):
            clear_history()

    st.title("🦾 Chatbot (news,law)")

    # Initialize the chatbot and load embeddings
    if "messages" not in st.session_state:
        with st.spinner("Initializing, please wait a moment!!!"):
            st.session_state.vector_store = load_embeddings()
            st.success("Finish!!!")
        st.session_state["messages"] = [{"role": "assistant", "content": "Tôi có thể giúp gì được cho bạn?"}]

    messages = st.session_state.messages
    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Please provide extra information",
        "on_submit": _submit_feedback,
    }

    for n, msg in enumerate(messages):
        st.chat_message(msg["role"]).write(msg["content"])

        if msg["role"] == "assistant" and n > 1:
            feedback_key = f"feedback_{int(n/2)}"

            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = None

            streamlit_feedback(
                **feedback_kwargs,
                key=feedback_key,
            )


    chat_history_placeholder = st.empty()
    if "history" not in st.session_state:
        st.session_state.history = []

    if prompt := st.chat_input():
        if "vector_store" in st.session_state:
            vector_store = st.session_state["vector_store"]
            
            q = prompt

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            with st.spinner("Thinking..."):
                response = ask_with_memory(vector_store, q, st.session_state.history)

            if len(st.session_state.history) >= chat_context_length:
                st.session_state.history = st.session_state.history[1:]

            st.session_state.history.append((q, response['answer']))

            chat_history_str = format_chat_history(st.session_state.history)

            msg = {"role": "assistant", "content": response['answer']}
            st.session_state.messages.append(msg)
            st.chat_message("assistant").write(msg["content"])

            # Display the feedback component after the chatbot responds
            feedback_key = f"feedback_{len(st.session_state.messages) - 1}"
            streamlit_feedback(
                **feedback_kwargs,
                key=feedback_key,
            )