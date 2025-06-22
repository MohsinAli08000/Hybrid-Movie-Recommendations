# 🎬 Hybrid Movie Recommendation System

This project presents a **Hybrid Movie Recommendation System** that combines both **Collaborative Filtering** and **Content-Based Filtering** techniques. The goal is to enhance the accuracy, personalization, and diversity of movie suggestions by leveraging the strengths of both approaches.

---

## 🔍 Project Overview

Recommender systems have become essential in today’s digital platforms, helping users navigate large volumes of content. However, single-method recommenders often suffer from limitations such as cold-start problems or a lack of personalization.

This project tackles these challenges by integrating:

- **Collaborative Filtering**, which recommends movies based on user-user or item-item similarity using rating patterns.
- **Content-Based Filtering**, which recommends movies by analyzing metadata like genres and titles.

The **hybrid approach** balances these two methods, ensuring users receive relevant suggestions even when limited data is available.

---

## 🎯 Objectives

- Build a robust movie recommendation system using MovieLens dataset
- Integrate collaborative and content-based filtering
- Provide real-time suggestions through a user-friendly web interface
- Address cold-start and sparsity problems inherent in recommendation systems

---

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy, Scikit-learn**
- **Streamlit** – for interactive web interface
- **MovieLens Dataset (100k, .dat format)**

---

## 💡 Methodology

### 1. **Data Preprocessing**
- Loaded and parsed `movies.dat`, `ratings.dat`, and `users.dat` files
- Merged datasets to associate movie genres and user preferences with ratings

### 2. **Content-Based Filtering**
- Vectorized genres using `TfidfVectorizer`
- Calculated cosine similarity between movies based on genre vectors
- Recommended movies similar to the user’s previously rated items

### 3. **Collaborative Filtering**
- Created a user-item matrix from ratings
- Utilized cosine similarity to find similar users
- Predicted ratings using neighbors' preferences

### 4. **Hybrid Recommender**
- Combined the content and collaborative recommendations using weighted average scores
- Allowed configurable weights to control balance between methods

### 5. **Web Interface with Streamlit**
- Developed an interactive web app using Streamlit
- Users select a movie, and receive hybrid recommendations in real-time
- Displayed top N recommended movies with titles and genres

---

## 📊 Results

- Successfully delivered personalized recommendations for a wide range of users
- The hybrid model outperformed individual methods in terms of diversity and relevance
- Solved cold-start problem partially through content-based layer

---

## 🙋 Author

Mohsin Ali
Data Scientist | AI Consultant
GitHub: @mohsinali08000
