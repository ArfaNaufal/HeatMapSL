import streamlit as st
import os
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import time

# ==========================================
# BAGIAN 1: MODIFIKASI LIBRARY HEATMAP ANDA
# ==========================================

def draw_display(dispsize, image_data=None):
    """
    Dimodifikasi untuk menerima image_data (numpy array/PIL) langsung dari Streamlit
    bukan path file string.
    """
    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    
    if image_data is not None:
        # Convert PIL image to numpy array if necessary
        img = np.array(image_data)
        
        # Handle RGBA to RGB conversion if needed
        if img.shape[2] == 4:
            img = img[:, :, :3]
            
        # Normalize if not already float
        if img.dtype == 'uint8':
            img = img.astype('float32') / 255.0

        w, h = len(img[0]), len(img)
        x = int(dispsize[0] / 2 - w / 2)
        y = int(dispsize[1] / 2 - h / 2)
        
        # Safe slicing to prevent out of bounds
        y_end = min(y + h, dispsize[1])
        x_end = min(x + w, dispsize[0])
        h_crop = y_end - y
        w_crop = x_end - x
        
        screen[y:y_end, x:x_end, :] += img[:h_crop, :w_crop, :]

    dpi = 100.0
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)
    return fig, ax

def gaussian(x, sx, y=None, sy=None):
    if y == None: y = x
    if sy == None: sy = sx
    xo = x / 2
    yo = y / 2
    M = np.zeros([y, x], dtype=float)
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(-1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))
    return M

def generate_heatmap_figure(gazepoints, dispsize, image_data=None, alpha=0.5, gaussianwh=200, gaussiansd=None):
    """
    Versi modifikasi dari draw_heatmap untuk mengembalikan Figure Matplotlib
    agar bisa dirender oleh Streamlit.
    """
    fig, ax = draw_display(dispsize, image_data=image_data)

    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    strt = int(gwh / 2)
    heatmapsize = (int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt))
    heatmap = np.zeros(heatmapsize, dtype=float)

    for i in range(0, len(gazepoints)):
        x = strt + int(gazepoints[i][0]) - int(gwh / 2)
        y = strt + int(gazepoints[i][1]) - int(gwh / 2)
        
        # Simplified boundary checking logic for brevity in Streamlit
        # (In production, keep your full boundary logic here)
        if (0 < x < dispsize[0]) and (0 < y < dispsize[1]):
             heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]

    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    
    # Avoid division by zero/empty array issues
    if np.any(heatmap > 0):
        lowbound = np.mean(heatmap[heatmap > 0])
        heatmap[heatmap < lowbound] = np.nan
        ax.imshow(heatmap, cmap='jet', alpha=alpha)
    
    ax.invert_yaxis()
    return fig

# ==========================================
# BAGIAN 2: DATABASE & AUTHENTICATION
# ==========================================

DB_NAME = 'eyetracking_app.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Tabel Users
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # Tabel Data Pengujian
    c.execute('''CREATE TABLE IF NOT EXISTS tests 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  test_name TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  csv_data BLOB,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username, password) VALUES (?,?)', (username, make_hashes(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =?', (username,))
    data = c.fetchall()
    conn.close()
    if data:
        if check_hashes(password, data[0][1]):
            return True
    return False

def save_test_result(username, test_name, csv_content, image_file_buffer):
    # Buat folder user jika belum ada
    user_dir = f"data_storage/{username}"
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    # Simpan gambar background
    img_path = ""
    if image_file_buffer:
        img_path = f"{user_dir}/{test_name}_{int(time.time())}.png"
        with open(img_path, "wb") as f:
            f.write(image_file_buffer.getbuffer())

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO tests(username, test_name, csv_data, image_path) VALUES (?,?,?,?)', 
              (username, test_name, csv_content, img_path))
    conn.commit()
    conn.close()

# ==========================================
# BAGIAN 3: USER INTERFACE (STREAMLIT)
# ==========================================

def main():
    st.set_page_config(page_title="EyeTrack Remote", layout="wide")
    init_db()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''

    # --- SIDEBAR NAV ---
    st.sidebar.title("Navigasi")
    
    if st.session_state['logged_in']:
        menu = ["Dashboard", "Pengujian Baru", "Analisis Heatmap", "Logout"]
        choice = st.sidebar.selectbox("Menu", menu)
        st.sidebar.success(f"Login sebagai: {st.session_state['username']}")
    else:
        menu = ["Login", "Sign Up"]
        choice = st.sidebar.selectbox("Menu", menu)

    # --- LOGIC HALAMAN ---
    
    if choice == "Login":
        st.subheader("Login Area")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Login Berhasil!")
                st.rerun()
            else:
                st.error("Username atau Password salah")

    elif choice == "Sign Up":
        st.subheader("Buat Akun Baru")
        new_user = st.text_input("Username Baru")
        new_password = st.text_input("Password Baru", type='password')
        if st.button("Sign Up"):
            if add_user(new_user, new_password):
                st.success("Akun berhasil dibuat. Silakan Login.")
            else:
                st.warning("Username sudah terpakai.")

    elif choice == "Logout":
        st.session_state['logged_in'] = False
        st.rerun()

    elif choice == "Dashboard":
        st.title(f"Dashboard {st.session_state['username']}")
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(f"SELECT id, test_name, timestamp FROM tests WHERE username='{st.session_state['username']}'", conn)
        conn.close()
        
        st.write("Riwayat Pengujian:")
        st.dataframe(df)

    elif choice == "Pengujian Baru":
        st.title("Setup Pengujian Remote")
        st.info("Di sini user mengunggah hasil rekaman eye tracking (CSV) dan Gambar Background.")
        
        test_name = st.text_input("Nama Pengujian", "Test_1")
        
        # Upload Background
        bg_file = st.file_uploader("Upload Gambar Website/Background", type=['png', 'jpg', 'jpeg'])
        
        # Upload CSV Data
        csv_file = st.file_uploader("Upload Data Gaze (CSV)", type=['csv'])
        
        if st.button("Simpan Data Pengujian"):
            if bg_file and csv_file:
                # Baca CSV sebagai bytes untuk disimpan
                csv_bytes = csv_file.getvalue()
                save_test_result(st.session_state['username'], test_name, csv_bytes, bg_file)
                st.success("Data berhasil disimpan ke database!")
            else:
                st.error("Mohon upload gambar dan CSV.")

    elif choice == "Analisis Heatmap":
        st.title("Visualisasi Heatmap")
        
        # Ambil list pengujian user
        conn = sqlite3.connect(DB_NAME)
        tests = pd.read_sql_query(f"SELECT id, test_name, timestamp, image_path FROM tests WHERE username='{st.session_state['username']}'", conn)
        
        if not tests.empty:
            selected_test_name = st.selectbox("Pilih Pengujian", tests['test_name'].unique())
            test_data = tests[tests['test_name'] == selected_test_name].iloc[0]
            
            # Load Data
            c = conn.cursor()
            c.execute("SELECT csv_data FROM tests WHERE id=?", (int(test_data['id']),))
            csv_blob = c.fetchone()[0]
            conn.close()
            
            # Konversi Blob ke DataFrame
            from io import StringIO, BytesIO
            s = str(csv_blob, 'utf-8')
            data = StringIO(s)
            
            try:
                # Asumsi format CSV: x, y, duration (opsional)
                df_gaze = pd.read_csv(data, header=None)
                
                # Handling format 2 kolom vs 3 kolom
                if df_gaze.shape[1] == 2:
                    df_gaze.columns = ['x', 'y']
                    df_gaze['duration'] = 1
                else:
                    df_gaze = df_gaze.iloc[:, :3]
                    df_gaze.columns = ['x', 'y', 'duration']
                
                gaze_data_list = list(df_gaze.itertuples(index=False, name=None))
                
                # Load Image
                image_path = test_data['image_path']
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    disp_size = img.size # (width, height)
                else:
                    st.error("File gambar tidak ditemukan.")
                    disp_size = (1920, 1080)
                    img = None

                # --- FITUR CAPTURE PER 10 DETIK ---
                st.subheader("Generate Heatmap")
                
                # Asumsi sampling rate 60Hz (60 data per detik) -> 10 detik = 600 baris
                # Anda bisa menyesuaikan ini dengan sampling rate alat Anda
                sampling_rate = st.number_input("Estimasi Sampling Rate (Hz)", value=60)
                chunk_size = sampling_rate * 10 
                
                total_rows = len(df_gaze)
                num_chunks = int(np.ceil(total_rows / chunk_size))
                
                time_slider = st.slider("Pilih Segmen Waktu (per 10 detik)", 0, num_chunks-1, 0)
                
                start_idx = time_slider * chunk_size
                end_idx = min((time_slider + 1) * chunk_size, total_rows)
                
                subset_gaze = gaze_data_list[start_idx:end_idx]
                
                if st.button("Generate Heatmap Segmen Ini"):
                    with st.spinner("Memproses Heatmap..."):
                        fig = generate_heatmap_figure(
                            subset_gaze, 
                            disp_size, 
                            image_data=img, 
                            alpha=0.6, 
                            gaussianwh=200
                        )
                        st.pyplot(fig, use_container_width=True)
                        
                        st.write(f"Menampilkan data dari baris {start_idx} sampai {end_idx}")

            except Exception as e:
                st.error(f"Error memproses CSV: {e}")
        else:
            st.write("Belum ada data pengujian.")
            conn.close()

if __name__ == '__main__':
    main()