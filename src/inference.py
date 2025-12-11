import streamlit as st
from huggingface_hub import InferenceClient

# Ensure persistent client storage
if "hf_client" not in st.session_state:
    st.session_state["hf_client"] = None


def init_model(token, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Initialize the Llama model once and store inside Streamlit session_state.
    """
    st.session_state["hf_client"] = InferenceClient(model=model_id, token=token)
    print("HF client initialized.")


def generate_with_context(query, context, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Generate answer grounded in context using chat_completion.
    """
    client = st.session_state.hf_client

    if client is None:
        raise RuntimeError("Model not initialized. Call init_model(token) first.")

    messages = [
        {"role": "system", "content": (
            "You are an AI assistant trained to answer questions strictly based on "
            "the Occupational Safety, Health and Working Conditions Code, 2020."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = client.chat_completion(
        model=model_id,
        messages=messages,
        max_tokens=400,
        temperature=0.2,
        top_p=0.9,
    )

    return response.choices[0].message["content"]
