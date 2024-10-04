from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import os
import csv
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import insightface
import matplotlib.pyplot as plt
from typing import List

app = FastAPI()

# Khởi tạo mô hình AI
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)  # ctx_id=-1 để sử dụng CPU

# Hàm đọc embedding từ file CSV
def load_embeddings(embedding_dir):
    database = []
    for csv_file in os.listdir(embedding_dir):
        csv_path = os.path.join(embedding_dir, csv_file)
        movie_id = os.path.basename(csv_file).split('_faces.csv')[0]
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Bỏ qua header
            for row in reader:
                img_name = row[0]
                embedding = np.array(eval(row[1]))
                database.append((movie_id, img_name, embedding))
    return database

# Hàm tính độ đo cosine similarity
def find_similar_faces(image_data, database, top_n=10):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []

    faces = model.get(img)
    if len(faces) == 0:
        return []

    user_embedding = faces[0].embedding
    similarities = []
    for movie_id, img_name, db_embedding in database:
        similarity = cosine_similarity([user_embedding], [db_embedding])[0][0]
        similarities.append((movie_id, img_name, similarity))

    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
    return similarities[:top_n]

# Hàm tạo HTML cho các ảnh giống nhất
def generate_html_for_images(top_similar_faces, image_base_dir):
    html_output = ""
    if not os.path.exists(image_base_dir):
        return f"<p>Directory {image_base_dir} does not exist.</p>"
    
    for i, (movie_id, img_name, similarity) in enumerate(top_similar_faces):
        img_path = os.path.join(image_base_dir, movie_id, img_name)
        if not os.path.exists(img_path):
            continue
        img_url = f"/ml-20m/image/{movie_id}/{img_name}"
        html_output += f'<div style="display:inline-block; margin:10px;">'
        html_output += f'<img src="{img_url}" width="200" />'
        html_output += f'<p>{movie_id} - {similarity:.2f}</p>'
        html_output += '</div>'
    return html_output

@app.post("/predict", response_class=HTMLResponse)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File không phải là hình ảnh.")

    image_data = await file.read()

    embedding_dir = 'embedding_face_movie'
    database = load_embeddings(embedding_dir)

    top_similar_faces = find_similar_faces(image_data, database, top_n=10)
    if not top_similar_faces:
        return "<p>Không tìm thấy khuôn mặt hoặc không có kết quả tương đồng.</p>"

    image_base_dir = 'ml-20m/image'
    html_output = generate_html_for_images(top_similar_faces, image_base_dir)
    return html_output
