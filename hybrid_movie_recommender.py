# Step 2.0 – Import libraries
import pandas as pd

# Step 2.1 – Load datasets with correct encoding
movies = pd.read_csv('movies.dat', sep='::', engine='python', 
                     names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')

ratings = pd.read_csv('ratings.dat', sep='::', engine='python', 
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')

users = pd.read_csv('users.dat', sep='::', engine='python', 
                    names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='ISO-8859-1')