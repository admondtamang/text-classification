import pandas as pd
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
# Visualuzation
import matplotlib.pyplot as plt
from datasets import load_dataset

import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

dataset = load_dataset("okite97/news-data")

## Load data from CSV with a specified encoding
df= pd.DataFrame(dataset['train'])

##  Data cleaning
# filter only the columns we need i.e. sports, health, business
df = df[df['Category'].isin(['sports', 'health', 'business'])]
# Remove duplicate rows
df = df.drop_duplicates()
# Remove rows with missing values in the Title and Excerpt columns
df = df.dropna(subset=['Title', 'Excerpt'])


## Plotting in bar diagram
# df['Category'].value_counts().plot(kind='bar')
# plt.title('Number of articles per Category', size=27, pad=27)
# plt.show()


## TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Excerpt'])
 
## Apply K-means clustering
k = 3  # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(tfidf_matrix)

## Map cluster labels to categories
category_map = {
    0: 'business',
    1: 'sports',
    2: 'health'
}

## Function to classify the input document
def classify_document():
    input_text = text_area.get("1.0", tk.END).strip()
    if input_text:
        new_document_tfidf = tfidf_vectorizer.transform([input_text.lower()])
        predicted_cluster = kmeans.predict(new_document_tfidf)
        predicted_category = category_map[predicted_cluster[0]]
        messagebox.showinfo("Classification Result", f"The document belongs to category: {predicted_category}")
    else:
        messagebox.showwarning("Empty Input", "Please enter a document to classify.")


if __name__ == '__main__':    
    ## User interface 
    window = tk.Tk()
    window.title("Document Classification")

    # Create a text area for input
    text_area = scrolledtext.ScrolledText(window, width=50, height=10)
    text_area.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

    # Create a button to classify the document
    classify_button = tk.Button(window, text="Classify", command=classify_document)
    classify_button.grid(row=1, column=0, padx=10, pady=10)

    # Create a button to clear the input
    clear_button = tk.Button(window, text="Clear", command=lambda: text_area.delete("1.0", tk.END))
    clear_button.grid(row=1, column=1, padx=10, pady=10)

    ## Start the GUI event loop
    window.mainloop()
