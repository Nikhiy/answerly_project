import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding = embedding_model.embed_query("This is a test sentence to check the embedding model.")

st.write(embedding)