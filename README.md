# 🦚 Klasifikasi Batik Indonesia 🦚

Selamat datang di website **Klasifikasi Batik**! 🇮🇩

Website ini dibuat untuk mengenalkan dan mengklasifikasikan berbagai motif batik khas Indonesia menggunakan teknologi AI. Dengan tampilan modern, responsif, dan nuansa batik yang hangat, aplikasi ini cocok untuk edukasi, pelestarian budaya, maupun sekadar eksplorasi keindahan batik Nusantara.

---

## ✨ Fitur Utama
- 🎨 **Klasifikasi Gambar Batik**: Upload gambar batik dan dapatkan prediksi jenis batiknya (Betawi, Kawung, Megamendung, Parang, Sekar Jagad)
- 🖼️ **Preview Gambar**: Lihat preview gambar sebelum dan sesudah klasifikasi
- 📚 **Info & Video Edukasi**: Setiap batik punya pop-up penjelasan dan video YouTube
- 🔊 **Audio Relaksasi**: Musik menenangkan bertema batik, bisa diatur on/off
- 📱 **Tampilan Responsif**: Nyaman diakses di desktop maupun mobile
- 👤 **Profil Pembuat**: Kenali pembuat website ini

---

## 🚀 Cara Menjalankan
1. **Clone repo & install dependensi**
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan aplikasi**
   ```bash
   python app.py
   ```
3. **Akses di browser**
   - Beranda: [http://localhost:5000/](http://localhost:5000/)
   - Klasifikasi: [http://localhost:5000/klasifikasi](http://localhost:5000/klasifikasi)
   - Profil: [http://localhost:5000/profil](http://localhost:5000/profil)

---

## 🧩 Struktur Fitur
- **Beranda**: Penjelasan batik, daftar batik, pop-up info & video
- **Klasifikasi**: Upload gambar, preview, hasil prediksi
- **Profil**: Nama, kelas, alasan pembuatan, dan foto pembuat

---

## 📂 Struktur Folder Penting
- `model/` : Model AI `.h5` untuk klasifikasi batik
- `static/` : Gambar batik, audio, style.css, foto profil
- `templates/` : HTML untuk semua halaman

---

## 👨‍💻 Tentang Pembuat
Website ini dibuat oleh **Darmawan Suhara** (4IA28) sebagai tugas dan kontribusi untuk pelestarian budaya Indonesia melalui teknologi.

> "Batik adalah identitas bangsa. Mari lestarikan dan kenali lebih dalam!" 🦚

---

## 💡 Catatan
- Pastikan file model dan gambar sudah ada di folder `static/` dan `model/`.
- Website ini menggunakan Flask, Bootstrap, dan TensorFlow.

---

Terima kasih sudah berkunjung! 🌺
