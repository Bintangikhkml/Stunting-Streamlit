import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np

# === 1. Konfigurasi Halaman ===
st.set_page_config(
    page_title="Prediksi Stunting Pada Balita",
    page_icon="📊",
    layout="wide",
)

# === 2. Sidebar Menu ===
st.sidebar.image("rb_7753.png", use_column_width=False, width=150)
st.sidebar.title("📋 Menu Utama")
menu = st.sidebar.radio("Pilih Menu", ["Dashboard", "Prediksi", "Dataset", "Grafik Stunting"])

# Judul aplikasi
st.title("Prediksi Stunting Pada Balita")

# Membaca file Excel
uploaded_file = "D:\SKRIPSI\STREAMLIT 2\dataset _terbaru_2025.xlsx"
data = pd.read_excel(uploaded_file)

# Normalisasi nama kolom
data.columns = data.columns.str.strip().str.lower()  # Hapus spasi dan ubah ke huruf kecil

# Preproses data usia (konversi ke tahun desimal)
def parse_usia(usia):
    try:
        parts = usia.split(" - ")
        tahun = int(parts[0].split(" ")[0])
        bulan = int(parts[1].split(" ")[0])
        hari = int(parts[2].split(" ")[0])
        return tahun + bulan / 12 + hari / 365
    except:
        return None

if "usia" in data.columns:
    data["usia (tahun)"] = data["usia"].apply(parse_usia)

# === 3. Konten berdasarkan Menu yang Dipilih ===

# 1. Dashboard
if menu == "Dashboard":
    st.header("Dashboard")
    st.write("Stunting adalah kondisi dimana tinggi badan seorang anak lebih pendek dari rata-rata tinggi anak pada usia yang sama, yang disebabkan oleh kekurangan gizi kronis, infeksi berulang, dan faktor lingkungan yang kurang mendukung. Stunting umumnya terjadi pada 1.000 hari pertama kehidupan, yaitu dari konsepsi hingga usia 2 tahun, yang merupakan periode paling penting untuk pertumbuhan fisik dan perkembangan otak anak.")
    st.write("Menurut World Health Organization (WHO), stunting terjadi ketika tinggi badan anak berada di bawah dua standar deviasi dari median tinggi badan anak seumuran pada populasi referensi WHO. Ini berarti, anak dengan stunting memiliki perkembangan fisik yang tertunda dan tidak mampu tumbuh dengan baik dalam jangka waktu yang lama.")
    st.write("WHO mengembangkan standar referensi pertumbuhan global yang digunakan secara internasional, termasuk di Asia, untuk menilai status pertumbuhan anak-anak dari berbagai latar belakang etnis dan geografi. Referensi ini diperoleh dari anak-anak yang sehat, menyusui, dan tumbuh di lingkungan dengan gizi baik, yang mencakup anak-anak dari berbagai wilayah, termasuk Asia. Namun, ada perbedaan dalam asupan gizi, kebiasaan hidup, dan lingkungan sosial yang mungkin mempengaruhi hasil pengukuran di setiap negara atau wilayah. Oleh karena itu, meskipun standar ini digunakan secara luas, dalam beberapa kasus, setiap negara bisa mengadaptasi pedoman ini berdasarkan kondisi lokal dan faktor-faktor khusus yang berhubungan dengan pertumbuhan anak di daerah tersebut.")
    st.image("—Pngtree—small child stunting child malnutrition_12257344.png", use_column_width=False, width=300)
# 2. Prediksi
elif menu == "Prediksi":
    st.header("Prediksi Stunting")
    st.write("Formulir untuk memulai prediksi stunting pada balita berdasarkan data yang diinput.")
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-Laki", "Perempuan"])
    usia = st.selectbox("Usia (Dalam Bulan)", list(range(61)))
    berat_badan_saat_ini = st.selectbox("Berat Badan Saat Ini (kg)", [round(i * 0.1, 1) for i in range(501)])
    tinggi_badan_saat_ini = st.selectbox("Tinggi Badan Saat Ini (cm)", [round(i * 0.1, 1) for i in range(1501)])
    liLA = st.selectbox("Lingkar Lengan Atas (cm)", [round(i * 0.1, 1) for i in range(101)])

    # Tombol Prediksi
    if st.button("Prediksi Stunting"):
        # Logika prediksi
        if usia < 24:  # 2 tahun pertama sangat penting
            if berat_badan_saat_ini < 7.0 or tinggi_badan_saat_ini < 70:
                hasil = "Pasien Terdiagnosa Stunting"
            else:
                hasil = "Pasien Tidak Terdiagnosa Stunting"
        elif usia >= 24 and usia <= 60:  # Usia 2-5 tahun
            if berat_badan_saat_ini < 10.0 or tinggi_badan_saat_ini < 90:
                hasil = "Pasien Terdiagnosa Stunting"
            else:
                hasil = "Pasien Tidak Terdiagnosa Stunting"
        else:
            hasil = "Usia tidak valid untuk prediksi stunting"

        st.subheader("Hasil Prediksi:")
        st.write(hasil)

# 3. Dataset
elif menu == "Dataset":
    st.header("Dataset")
    st.write("Halaman ini memungkinkan Anda untuk mengunggah dataset dalam format Excel dan melihat data yang diunggah.")
    
    # Upload dataset
    uploaded_file = st.file_uploader("Unggah file Excel:", type=["xlsx"])
    
    if uploaded_file:
        # Membaca dataset yang diunggah
        data = pd.read_excel(uploaded_file)
        st.write("**Dataset yang diunggah:**")
        st.dataframe(data)
        
        # Menangani nilai NaN
        data = data.dropna(subset=['JK', 'Berat', 'Tinggi', 'LiLA', 'Usia', 'Katagori Gizi'])  # Menghapus NaN dari fitur dan target
        data['Berat'] = data['Berat'].fillna(data['Berat'].mean())  # Mengganti NaN pada kolom 'Berat' dengan nilai rata-rata
        data['Tinggi'] = data['Tinggi'].fillna(data['Tinggi'].mean())  # Mengganti NaN pada kolom 'Tinggi' dengan nilai rata-rata
        data['LiLA'] = data['LiLA'].fillna(data['LiLA'].mean())  # Mengganti NaN pada kolom 'LiLA' dengan nilai rata-rata
        data['Usia'] = data['Usia'].fillna(data['Usia'].mean())  # Mengganti NaN pada kolom 'Usia' dengan nilai rata-rata
        
        # Mengambil kolom fitur dan target
        if 'Katagori Gizi' in data.columns:
            # Fitur
            X = data[['JK', 'Berat', 'Tinggi', 'LiLA', 'Usia']]  # Fitur yang digunakan
            # Target (label) - Gizi Baik vs Gizi Tidak Baik (misalnya)
            y = data['Katagori Gizi'].apply(lambda x: 1 if x == 'Gizi Baik' else 0)  # Konversi kategori gizi menjadi 1 (baik) dan 0 (tidak baik)
            
            # Pembagian data train dan test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Melatih model Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Melatih model Logistic Regression
            logreg_model = LogisticRegression(random_state=42)
            logreg_model.fit(X_train, y_train)
            
            # Memprediksi dengan kedua model
            y_pred_rf_no_fs = rf_model.predict(X_test)
            y_pred_logreg = logreg_model.predict(X_test)
            
            # Menghitung akurasi
            rf_accuracy = rf_model.score(X_test, y_test) * 100
            logreg_accuracy = logreg_model.score(X_test, y_test) * 100
            
            # Perbandingan Akurasi
            st.write("### Perbandingan Akurasi Model")
            models = ['Random Forest', 'Logistic Regression']
            accuracies = [rf_accuracy, logreg_accuracy]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(models, accuracies, color=['blue', 'orange'])
            for i, v in enumerate(accuracies):
                ax.text(i, v + 1, f"{v:.2f}%", ha='center')
            ax.set_title('Perbandingan Akurasi Model')
            ax.set_ylabel('Akurasi (%)')
            ax.set_ylim(0, 100)
            st.pyplot(fig)

        else:
            st.write("Kolom 'Katagori Gizi' tidak ditemukan dalam dataset.")


# 5. Grafik Stunting
elif menu == "Grafik Stunting":
    st.header("Grafik Stunting")
    st.write("Halaman ini menampilkan grafik distribusi stunting dan informasi terkait lainnya.")
    
    # Menampilkan gambar yang diunggah
    st.image('D:\SKRIPSI\Streamlit\perbandingan matrix model .png', caption='Perbandingan Metrik Model: Random Forest vs Logistic Regression')
    st.image('D:\SKRIPSI\Streamlit\split data selesai .png', caption='Pembagian Data Pelatihan dan Pengujian')
    st.image('D:\SKRIPSI\Streamlit\distribusi tinggi berdasarkan usia.png', caption='Distribusi Tinggi Badan Berdasarkan Usia (Bulan)')
    
    # Visualisasi distribusi Stunting
    if "usia (tahun)" in data.columns:
        st.write("**Distribusi Umur:**")
        fig, ax = plt.subplots()
        ax.hist(data["usia (tahun)"].dropna(), bins=10, edgecolor='black')
        ax.set_title("Distribusi Umur (Tahun)")
        ax.set_xlabel("Umur (Tahun)")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    if "berat" in data.columns:
        st.write("**Distribusi Berat:**")
        fig, ax = plt.subplots()
        ax.hist(data["berat"].dropna(), bins=10, edgecolor='black')
        ax.set_title("Distribusi Berat Badan")
        ax.set_xlabel("Berat (kg)")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    if "jk" in data.columns:
        st.write("**Distribusi Jenis Kelamin:**")
        gender_counts = data["jk"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(gender_counts.index, gender_counts.values, color=['blue', 'pink'])
        ax.set_title("Distribusi Jenis Kelamin")
        ax.set_xlabel("Jenis Kelamin")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)
    else:
        st.error("Kolom 'jk' tidak ditemukan.")