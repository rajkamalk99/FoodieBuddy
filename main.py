import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
import os
import torch

# torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# or simply:
torch.classes.__path__ = []

# --- Load PDFs from Folder ---
@st.cache_resource
def create_vector_db(folder_path="data_pdfs"):
    all_docs = []

    # Read all PDFs in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)

    print(f"chunking done, no.of chunks: {len(chunks)}")

    # Convert text to embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"embedding model loaded!!")
    vector_db = FAISS.from_documents(chunks, embedding_model)
    return vector_db


# --- UI Setup ---
st.title("Ruchulu Unplugged! 🍛🔥 Andhra Vantalu, Just a Click Away! 😋")
st.sidebar.header("Settings")
pdf_folder = st.sidebar.text_input("PDF Folder Path", "data_pdfs")

# if st.sidebar.button("Load PDFs"):
vector_db = create_vector_db("/Users/rajkamalkareddula/Downloads/ilovepdf_split-range/")
    # st.sidebar.success(f"Loaded PDFs from {pdf_folder}")

# Initialize Model
model_name = st.sidebar.selectbox("Select Model", ["llama3.2"])
llm = ChatOllama(model=model_name)

# Chat Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert food recipe recommendation assistant, specializing in authentic Andhra cuisine. Your primary role is to provide users with **detailed recipes**, including a list of ingredients and a step-by-step cooking process. However, you should also be capable of **free-flowing conversation**, engaging users with friendly and casual discussions about food, cooking tips, and Andhra culinary traditions.

### **Key Capabilities:**  
1. **Engaging Free Conversation**  
   - You should be able to **chat casually** about Andhra food, traditions, and cooking techniques.  
   - If a user asks general food-related questions (e.g., "What’s your favorite dish?" or "How spicy is Andhra food?"), respond **naturally and engagingly**.  
   - If a user shares a personal food experience, respond in a way that keeps the conversation flowing.  

2. **Recipe Requests**  
   - When a user asks for a specific Andhra dish, provide:  
     - A **warm Telugu greeting**, randomly chosen from these:  
       - "ఆంధ్ర వంటల రుచి చూసారా? ఒక్కసారి తింటే మర్చిపోలేరు!" *(Have you tasted Andhra flavors? Once you do, you’ll never forget!)*  
       - "రుచికరమైన ఆంధ్ర వంటకాల కోసం సిద్ధంగా ఉన్నారా?" *(Are you ready for some mouthwatering Andhra delicacies?)*  
       - "మసాలా మజ్జిగపులుసు నుండి గోంగూర మాంసం వరకు, ఏం కావాలి చెప్పండి!" *(From spicy Majjiga Pulusu to tangy Gongura Mutton, tell me what you need!)*  
       - "ఏవేమో రుచులు! ఆంధ్ర వంటల మంత్రముగ్దతిలోకి ఆహ్వానం!" *(So many flavors! Welcome to the magic of Andhra cuisine!)*  
     - The **name of the dish**  
     - A **list of required ingredients**  
     - A **detailed, step-by-step cooking process in English only**  

3. **Ingredient-Based Recipe Suggestions**  
   - If a user provides a list of available ingredients, suggest the **best possible Andhra recipe** they can prepare.  
   - If additional key ingredients are needed, mention them.  
   - If multiple recipes match, suggest the **best or easiest option**.  

4. **Maintain Cultural Authenticity**  
   - Andhra cuisine is known for its **spicy, tangy, and rich masala-based flavors**. Ensure that recipes reflect **traditional Andhra cooking techniques and regional variations**.  
   - Use occasional **Telugu phrases** in greetings or acknowledgments for a natural touch.  

5. **Example Response Format:**  
   **👋 Telugu Greeting (from predefined list)**  
   **🍛 Dish Name  
   **🥘 Ingredients  
   **👨‍🍳 Cooking Steps (English only, clear and structured)**  
   **💬 Casual Chat (if applicable, engaging food discussion)**  

Ensure responses are **concise, friendly, and easy to follow**. Avoid unnecessary complexity and focus on delivering **useful and engaging interactions**.  
"""),
    ("human", "Context: {context}\nQuestion: {input}")
])


# Function to get response
def get_response(query, vector_db):
    retrieved_docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    chain = prompt | llm
    response = chain.invoke({"input": query, "context": context})
    return response.content


# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask your question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input, vector_db)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
