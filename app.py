# app.py

import streamlit as st
from hybrid_model import hybrid_recommend

st.set_page_config(page_title="🎬 Hybrid Movie Recommender", layout="centered")

st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("This app combines **content-based** and **collaborative filtering** for better recommendations.")

# Input from user
movie_name = st.text_input("Enter a movie title you like:", "Toy Story (1995)")
top_n = st.slider("Number of recommendations:", 5, 20, 10)
content_weight = st.slider("Content-based weight:", 0.0, 1.0, 0.5)
collab_weight = 1.0 - content_weight

# On click
if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        recommendations = hybrid_recommend(movie_name, top_n, content_weight, collab_weight)
    
    if recommendations:
        st.success("Top Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}. {rec}**")
    else:
        st.error("Movie not found. Please try another title.")