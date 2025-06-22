# 🎬 Hybrid Movie Recommendation System

A hybrid movie recommender system that combines **content-based filtering** and **collaborative filtering** to generate personalized movie recommendations. Built with **Python**, **Pandas**, **Scikit-learn**, and deployed using **Streamlit**.

---

## 🚀 Features

- 🔄 **Hybrid Filtering**: Combines genre-based (content) and user-rating-based (collaborative) filtering.
- 📊 **Cosine Similarity**: Calculates similarity between movies and users.
- 🎛️ **Tunable Weights**: Adjustable sliders for content vs. collaborative influence.
- 💻 **Interactive UI**: Built using Streamlit for real-time recommendation generation.
- 📂 **Uses MovieLens-style Dataset**: Reads from `movies.dat`, `ratings.dat`, and `users.dat`.

---

## 🛠️ Tech Stack

- Python 3.x  
- Pandas  
- Scikit-learn  
- Streamlit

---

## 📂 Project Structure

├── app.py # Streamlit application
├── hybrid_model.py # Hybrid recommendation logic
├── hybrid_movie_recommender.ipynb # Jupyter Notebook version
├── requirements.txt # Project dependencies
├── data/
│ ├── movies.dat
│ ├── ratings.dat
│ └── users.dat
