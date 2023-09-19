import streamlit as st
import os
import streamlit.components.v1 as components
from io import BytesIO
import requests
import ast

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from bardapi import Bard
from typing import Any, List, Mapping, Optional

os.environ['_BARD_API_KEY'] = "aAhD1NyQqzeoXs8PclDOD_hvEI3N9uHnsn2F0isADM5FFwBfYxatJf1csSUTMo4TXLjOxA."

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
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
        response = Bard(token=os.environ['_BARD_API_KEY']).get_answer(prompt)['content']
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
    
@st.cache_data
def get_image(url):
    r = requests.get(url)
    return BytesIO(r.content)


# Define global variables
embeddings = None
index = None
QUESTION_PROMPT = None
qa = None
result = []

# Custom session state class for managing pagination
class SessionState:
    def __init__(self):
        self.page_index = 0  # Initialize page index
        self.database_loaded = False  # Initialize database loaded state
        self.all_results_displayed = False

# Create a session state object
session_state = SessionState()

# Define the search function outside of Search_Property
def display_search_results(result, start_idx, end_idx):
    if result:
        st.subheader("Search Results:")
        for idx in range(start_idx, end_idx):
            if idx >= len(result):
                break
            property_info = result[idx]
            st.markdown(f"**Result {idx + 1}**")
            
            # Display property information
            image_path_urls = property_info.metadata['Image URL']
            if image_path_urls is not None and not isinstance(image_path_urls, float):
                # Convert the string to a Python list
                imageUrls = ast.literal_eval(image_path_urls)

                # Now, imageUrls is a list of strings
                st.image(imageUrls[0],width=700)

            st.markdown(f"🏡 {property_info.metadata['Title']}")
            st.write(f"📍 Address: {property_info.metadata['Location']}")
            st.markdown(f"💰 Price: {property_info.metadata['Price']} VND | 📏 Size: {property_info.metadata['Area']}")
            st.markdown(f"📅 Published Date: {property_info.metadata['Time stamp']}") 
            col3, col4 = st.columns([2, 1]) 
            with col3: 
                with st.expander("Full Property Information"):
                    st.write(f"🏡 Property Title: {property_info.metadata['Title']}")
                    st.write(f"📏 Size: {property_info.metadata['Area']}")
                    st.write(f"🏢 Category: {property_info.metadata['Category']}")
                    st.write(f"📝 Description: {property_info.metadata['Description']}")
                    st.write(f"💰 Price: {property_info.metadata['Price']} VND")
                    st.write(f"📅 Date: {property_info.metadata['Time stamp']}")
                    st.write(f"📍 Address: {property_info.metadata['Location']}")
                    st.write(f"🆔 ID: {property_info.metadata['ID']}")
                    if 'Estate type' in property_info.metadata and property_info.metadata['Estate type'] is not None and not isinstance(property_info.metadata['Estate type'], float):
                        st.write(f"🏠 Housing Type: {property_info.metadata['Estate type']}")
                    if 'Email' in property_info.metadata and property_info.metadata['Email'] is not None and not isinstance(property_info.metadata['Email'], float):
                        st.write(f"✉️ Email: {property_info.metadata['Email']}")
                    if 'Mobile Phone' in property_info.metadata and property_info.metadata['Mobile Phone'] is not None and not isinstance(property_info.metadata['Mobile Phone'], float):
                        st.write(f"📞 Phone: {property_info.metadata['Mobile Phone']}")
                    if 'Certification status' in property_info.metadata and property_info.metadata['Certification status'] is not None and not isinstance(property_info.metadata['Certification status'], float):
                        st.write(f"🏆 Certification status: {property_info.metadata['Certification status']}")
                    if 'Direction' in property_info.metadata and property_info.metadata['Direction'] is not None and not isinstance(property_info.metadata['Direction'], float):
                        st.write(f"🧭 Direction: {property_info.metadata['Direction']}")
                    if 'Rooms' in property_info.metadata and property_info.metadata['Rooms'] is not None and not isinstance(property_info.metadata['Rooms'], float):
                        st.write(f"🚪 Rooms: {property_info.metadata['Rooms']}")
                    if 'Bedrooms' in property_info.metadata and property_info.metadata['Bedrooms'] is not None and not isinstance(property_info.metadata['Bedrooms'], float):
                        st.write(f"🛏️ Bedrooms: {property_info.metadata['Bedrooms']}")
                    if 'Kitchen' in property_info.metadata and property_info.metadata['Kitchen'] is not None and not isinstance(property_info.metadata['Kitchen'], float):
                        st.write(f"🍽️ Kitchen: {property_info.metadata['Kitchen']}")
                    if 'Living room' in property_info.metadata and property_info.metadata['Living room'] is not None and not isinstance(property_info.metadata['Living room'], float):
                        st.write(f"🛋️ Living room: {property_info.metadata['Living room']}")
                    if 'Bathrooms' in property_info.metadata and property_info.metadata['Bathrooms'] is not None and not isinstance(property_info.metadata['Bathrooms'], float):
                        st.write(f"🚽 Bathrooms: {property_info.metadata['Bathrooms']}")
                    if 'Front width' in property_info.metadata and property_info.metadata['Front width'] is not None and not isinstance(property_info.metadata['Front width'], float):
                        st.write(f"📐 Front width: {property_info.metadata['Front width']}")
                    if 'Floor' in property_info.metadata and property_info.metadata['Floor'] is not None and not isinstance(property_info.metadata['Floor'], float):
                        st.write(f"🧱 Floor: {property_info.metadata['Floor']}")
                    if 'Parking Slot' in property_info.metadata and property_info.metadata['Parking Slot'] is not None and not isinstance(property_info.metadata['Parking Slot'], float):
                        st.write(f"🚗 Parking Slot: {property_info.metadata['Parking Slot']}")
                    if 'Seller name' in property_info.metadata and property_info.metadata['Seller name'] is not None and not isinstance(property_info.metadata['Seller name'], float):
                        st.write(f"👤 Seller Name: {property_info.metadata['Seller name']}")
                    if 'Seller type' in property_info.metadata and property_info.metadata['Seller type'] is not None and not isinstance(property_info.metadata['Seller type'], float):
                        st.write(f"👨‍💼 Seller type: {property_info.metadata['Seller type']}")
                    if 'Seller Address' in property_info.metadata and property_info.metadata['Seller Address'] is not None and not isinstance(property_info.metadata['Seller Address'], float):
                        st.write(f"📌 Seller Address: {property_info.metadata['Seller Address']}")
                    if 'Balcony Direction' in property_info.metadata and property_info.metadata['Balcony Direction'] is not None and not isinstance(property_info.metadata['Balcony Direction'], float):
                        st.write(f"🌄 Balcony Direction: {property_info.metadata['Balcony Direction']}")
                    if 'Furniture' in property_info.metadata and property_info.metadata['Furniture'] is not None and not isinstance(property_info.metadata['Furniture'], float):
                        st.write(f"🛋️ Furniture: {property_info.metadata['Furniture']}")
                    if 'Toilet' in property_info.metadata and property_info.metadata['Toilet'] is not None and not isinstance(property_info.metadata['Toilet'], float):
                        st.write(f"🚽 Toilet: {property_info.metadata['Toilet']}")                

            with col4:
                st.empty()

            imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")
            image_path_urls = property_info.metadata['Image URL']
            if image_path_urls is not None and not isinstance(image_path_urls, float):
                # Convert the string to a Python list
                imageUrls = ast.literal_eval(image_path_urls)
                if len(imageUrls) > 1:
                    selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)
                    if selectedImageUrl is not None:
                        st.image(selectedImageUrl)

            # Add a divider after displaying property info
            st.markdown("<hr style='border: 2px solid white'>", unsafe_allow_html=True)  # Horizontal rule as a divider
            

                    

def Search_Property():
    global embeddings, index, result, QUESTION_PROMPT, qa

    st.title("🏘️ Property Search ")
    # Load data and create the search
    if not session_state.database_loaded:
        st.info("Loading database... This may take a moment.")
        embeddings = SentenceTransformerEmbeddings(model_name="keepitreal/vietnamese-sbert")
        # Create a Chroma object with persistence
        db = Chroma(persist_directory="./chroma_index_1", embedding_function=embeddings)
        # Get documents from the database
        db.get()
        llm=BardLLM()
        qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k":4}),
        return_source_documents=True)
        question_template = """
        Context: You are a helpful and informative bot that answers questions posed below using page_content information from real estate documents.
        Do not create your own answer, just answer using page_content and metadata information from related documents in Vietnamese.
        Be sure to respond in a complete sentence, being comprehensive, including all metadata information.
        Imagine you're talking to a friend and use natural language and phrasing.
        You can only use Vietnamese do not use other languages.

        QUESTION: '{question}'

        ANSWER:
        """
        QUESTION_PROMPT = PromptTemplate(
            template=question_template, input_variables=["question"]
        )   
        session_state.database_loaded = True

    if session_state.database_loaded:
        col1, col2 = st.columns([2, 1])  # Create a two-column layout

        with col1:
            query = st.text_input("Enter your property search query:")
            search_button = st.button("Search", help="Click to start the search")

            if search_button:
                with st.spinner("Searching..."):
                    if query is not None:  # Check if model_embedding is not None
                        qa.combine_documents_chain.llm_chain.prompt = QUESTION_PROMPT
                        qa.combine_documents_chain.verbose = True
                        qa.return_source_documents = True
                        results = qa({"query":query,})
                        result = results["source_documents"]
                        session_state.page_index = 0  # Reset page index when a new search is performed
                    
        with col2:
            if len(result) > 0:
                st.write(f'Total Results: {len(result)} properties found.')  # Display "Total Results" in the second column

        if result:   
            N = 5 
            prev_button, next_button = st.columns([4,1])
            last_page = len(result) // N

            
            # Update page index based on button clicks
            if prev_button.button("Previous", key="prev_button"):
                if session_state.page_index - 1 < 0:
                    session_state.page_index = last_page
                else:
                    session_state.page_index -= 1

            if next_button.button("Next", key="next_button"):
                if session_state.page_index > last_page:
                    session_state.page_index = 0
                else:
                    session_state.page_index += 1

            # Calculate the range of results to display (5 properties at a time)
            start_idx = session_state.page_index * N 
            end_idx = (1 + session_state.page_index) * N

            # Display results for the current page
            display_search_results(result, start_idx, end_idx)
