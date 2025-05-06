#Import the required libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import requests
from io import BytesIO

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv('imdb_top_1000.csv')
df = df[['Series_Title', 'Overview', 'Poster_Link']].dropna()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_plot'] = df['Overview'].apply(clean_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_plot'])

def recommend_from_story(input_story, df, tfidf_matrix, top_n=5):
    clean_input = clean_text(input_story)
    input_vector = vectorizer.transform([clean_input])
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = [cosine_similarities[i] for i in top_indices]
    return results[['Series_Title', 'Overview', 'Poster_Link', 'similarity_score']]

#Create the GUI using Tkinter
window = tk.Tk()
window.title("🎥 Movie Recommendation System (Storyline Based)")
window.geometry("900x700")
window.configure(bg="#2c3e50")
window.rowconfigure(1, weight=1)
window.rowconfigure(3, weight=8)
window.columnconfigure(0, weight=1)

label = tk.Label(window, text="Enter a movie storyline:", font=("Helvetica", 13, "bold"),
                 fg="white", bg="#2c3e50")
label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

entry = tk.Text(window, height=5, wrap=tk.WORD, font=("Arial", 11), bg="#ecf0f1", fg="#2c3e50")
entry.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

button = tk.Button(window, text="🎯 Get Recommendations", command=lambda: show_recommendations(),
                   bg="#3498db", fg="white", font=("Arial", 12, "bold"))
button.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

canvas_frame = tk.Frame(window, bg="#ecf0f1")
canvas_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
canvas_frame.rowconfigure(0, weight=1)
canvas_frame.columnconfigure(0, weight=1)

canvas = tk.Canvas(canvas_frame, bg="#ecf0f1")
scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#ecf0f1")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

poster_images = []  # Keep a reference to avoid garbage collection

def show_recommendations():
    user_input = entry.get("1.0", tk.END).strip()
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    if not user_input:
        tk.Label(scrollable_frame, text="⚠️ Please enter a movie storyline.",
                 font=("Arial", 12), fg="red", bg="#ecf0f1").pack(pady=10)
        return

    recommendations = recommend_from_story(user_input, df, tfidf_matrix)

    for _, row in recommendations.iterrows():
        frame = tk.Frame(scrollable_frame, bg="#ecf0f1", pady=10)
        frame.pack(fill="x", padx=10)

        try:
            response = requests.get(row['Poster_Link'], timeout=5)
            img_data = response.content
            img = Image.open(BytesIO(img_data))
            img = img.resize((100, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            poster_images.append(photo)  # store reference

            poster_label = tk.Label(frame, image=photo, bg="#ecf0f1")
            poster_label.pack(side="left", padx=10)
        except:
            poster_label = tk.Label(frame, text="📷 No Image", bg="#ecf0f1", width=15, height=8)
            poster_label.pack(side="left", padx=10)

        text_frame = tk.Frame(frame, bg="#ecf0f1")
        text_frame.pack(side="left", fill="both", expand=True)

        tk.Label(text_frame, text=row['Series_Title'], font=("Arial", 13, "bold"),
                 fg="#2980b9", bg="#ecf0f1").pack(anchor="w")
        tk.Label(text_frame, text=f"Similarity Score: {row['similarity_score']:.2f}",
                 font=("Arial", 10), fg="#7f8c8d", bg="#ecf0f1").pack(anchor="w")
        tk.Label(text_frame, text=row['Overview'], wraplength=650, justify="left",
                 font=("Arial", 11), bg="#ecf0f1", fg="#2c3e50").pack(anchor="w")

window.mainloop()
