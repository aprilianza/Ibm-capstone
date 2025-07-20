import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
)

system_context = """
Kamu adalah asisten pribadi yang membantu user dengan berbagai tugas sehari-hari.
Berikan jawaban yang sopan, ramah, dan bantu user dengan efisien. Jika tidak yakin, katakan dengan jujur.
"""

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False,
)

st.set_page_config(page_title="Asisten Pribadi Anda", page_icon="🤖")
st.title("🧑‍💼 Personal Assistant")

user_input = st.chat_input("Tanyakan apa saja pada asisten pribadimu...")

if user_input:
    # Append to conversation
    memory.chat_memory.add_user_message(user_input)
    response = llm.invoke(system_context + "\n" + memory.buffer + f"\nUser: {user_input}")
    memory.chat_memory.add_ai_message(response.content)

    st.session_state.setdefault("chat_history", []).append(("user", user_input))
    st.session_state.chat_history.append(("assistant", response.content))

# --- Display chat history ---
if "chat_history" in st.session_state:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)
