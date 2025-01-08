import streamlit as st
from phi.agent import Agent
from phi.model.openai import OpenAIChat
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY

# Page configuration
st.set_page_config(
    page_title="Text Summary AI Agent",
    page_icon="📝",
    #layout="wide"
)

st.title("Agentic ChatBOT")
st.header("Powered by OpenAI 🤖")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Q&A chatbot",
        model=OpenAIChat(id="gpt-4o-mini"),
        markdown=True,
    )

# Initialize the agent
text_summarizer_agent = initialize_agent()

# Text area for user to enter long text
user_text = st.text_area(
    "Shoot your question here:",
    placeholder="I can answer anything Sagar trained me on",
    help="For any help call Sagar."
)

if st.button("Enter"):
    if not user_text:
        st.warning("Ask your question")
    else:
        try:
            with st.spinner("Getting the answer for you..."):
                # Create prompt for summarization
                summary_prompt = f"Answer the user question:\n{user_text}"

                # AI agent processing
                response = text_summarizer_agent.run(summary_prompt)

            # Display the result
            st.subheader("Response")
            st.markdown(response.content)

        except Exception as error:
            st.error(f"An error occurred during while getting the answer: {error}")
else:
    st.info("Sagar's Personalised ChatBot")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 70px;
    
    </style>
    """,
    unsafe_allow_html=True
)