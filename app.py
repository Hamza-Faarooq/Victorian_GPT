import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. The Aesthetic Configuration
st.set_page_config(page_title="VictorianGPT Archives", page_icon="🎩", layout="centered")
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>The Victorian Archives</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>Converse with a 19th-century scholar, grounded in classical literature.</p>", unsafe_allow_html=True)
st.divider()

# 2. Awaken the Machinery (Cached to prevent redundant loading)
@st.cache_resource
def load_heavy_machinery():
    print("Stoking the furnaces...")
    # Load ChromaDB Memory
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Load 4-bit Quantized Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = PeftModel.from_pretrained(base_model, "./victorian_adapter")
    
    model.eval()
    model.config.use_cache = True
    return tokenizer, model, vectorstore

tokenizer, model, vectorstore = load_heavy_machinery()

# 3. The Conversation Ledger
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Greetings. How may I be of service to you this day?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. The Interaction Loop
if prompt := st.chat_input("Pen your inquiry here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Consulting the historical texts..."):
            # RAG Retrieval
            docs = vectorstore.similarity_search(prompt, k=1)
            context_text = "\n".join([d.page_content for d in docs])
            
            system_instruction = (
                "You are a sophisticated, brooding scholar from the late 19th century. "
                "Speak in elegant, gothic prose, employing metaphors and philosophical observations."
            )
            
            formatted_prompt = f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n<|im_start|>user\n[Background Knowledge]\n{context_text}\n[User Query]\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.85,
                    repetition_penalty=1.15, # Crucial penalty to prevent context copy-pasting
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            reply = full_output.split("assistant\n")[-1].strip()
            
            st.markdown(reply)
            
            # Display Sources
            with st.expander("Examine the Source Texts"):
                for doc in docs:
                    st.caption(f"**From {doc.metadata['source']}:** {doc.page_content}")
            
            st.session_state.messages.append({"role": "assistant", "content": reply})
