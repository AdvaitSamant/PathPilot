import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return tokenizer, model

tokenizer, model = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 TinyLlama Chatbot")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Prepare input
        messages = [{"role": "user", "content": user_input}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate response
        with st.spinner("TinyLlama is thinking..."):
            outputs = model.generate(**inputs, max_new_tokens=1000)
            bot_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

        # Save bot response
        st.session_state.messages.append({"role": "bot", "content": bot_response})

        # Display latest bot response
        st.markdown(f"**Bot:** {bot_response}")
