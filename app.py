from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load model sekali saat start
MODEL_PATH = os.path.join('model', 'BatikDetection.h5')
model = load_model(MODEL_PATH)

# Daftar label batik sesuai urutan output model dan nama file di index.html
LABELS = [
    'Batik Betawi',
    'Batik Kawung',
    'Batik Megamendung',
    'Batik Parang',
    'Batik Sekar Jagad'
]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Ubah sesuai input model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi jika model pakai ini
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    result = None
    img_url = None
    if request.method == 'POST':
        if 'file' not in request.files:
            result = 'Tidak ada file yang diupload.'
        else:
            file = request.files['file']
            if file.filename == '':
                result = 'Tidak ada file yang dipilih.'
            else:
                temp_dir = os.path.join('static', 'temp')
                os.makedirs(temp_dir, exist_ok=True)
                filepath = os.path.join(temp_dir, file.filename)
                file.save(filepath)
                img = preprocess_image(filepath)
                preds = model.predict(img)[0]
                idx = np.argmax(preds)
                label = LABELS[idx]
                result = f"{label}"
                img_url = f"/static/temp/{file.filename}"
                # Tidak menghapus file agar bisa preview setelah klasifikasi
    return render_template('klasifikasi.html', result=result, img_url=img_url)

@app.route('/profil')
def profil():
    return render_template('profil.html')

if __name__ == '__main__':
    app.run(debug=True) 