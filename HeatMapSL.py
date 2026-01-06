import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import base64  # <--- TAMBAHAN IMPORT
from PIL import Image

# ==========================================
# 1. KONFIGURASI WEBGAZER (JAVASCRIPT)
# ==========================================
def get_webgazer_html(img_base64=None):
    # Logika untuk menampilkan gambar atau pesan placeholder
    if img_base64:
        display_content = f"""
            <div style="text-align: center; margin-bottom: 10px;">
                <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; max-height: 100%; border: 2px solid #ddd; border-radius: 4px;">
                <p style="font-size: 12px; color: grey;">Lihat gambar di atas selama perekaman</p>
            </div>
        """
    else:
        display_content = """
            <div style="text-align: center; padding: 20px; border: 2px dashed #ccc; color: #666;">
                <p>Gambar target belum diupload.</p>
            </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://webgazer.cs.brown.edu/webgazer.js"></script>
        <style>
            body {{ font-family: sans-serif; margin: 0; padding: 10px; }}
            .calibration-dot {{
                position: fixed; width: 20px; height: 20px;
                background: red; border-radius: 50%; opacity: 0.7;
                cursor: pointer; z-index: 9999; display: none;
            }}
            .btn {{
                padding: 10px 20px; background: #4CAF50; color: white;
                border: none; cursor: pointer; border-radius: 5px; margin: 5px;
            }}
            .btn-stop {{ background: #f44336; }}
            #status {{ margin-top: 10px; font-weight: bold; text-align: center; }}
            textarea {{ width: 100%; height: 50px; margin-top: 10px; display: none; }} 
            #controls {{ text-align: center; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h3 style="text-align: center;">Webcam Eye Tracker</h3>
        
        {display_content}
        
        <div id="controls">
            <button class="btn" onclick="startCalibration()">1. Mulai Kalibrasi</button>
            <button class="btn" onclick="startRecording()" id="recBtn" disabled>2. Mulai Rekam</button>
            <button class="btn btn-stop" onclick="stopRecording()" id="stopBtn" disabled>3. Stop & Copy</button>
        </div>
        <div id="status">Status: Upload gambar -> Kalibrasi -> Rekam</div>
        
        <textarea id="outputData" placeholder="Data JSON akan muncul di sini..." readonly></textarea>

        <div id="cal1" class="calibration-dot" style="top:5%; left:5%" onclick="calClick(this)"></div>
        <div id="cal2" class="calibration-dot" style="top:5%; right:5%" onclick="calClick(this)"></div>
        <div id="cal3" class="calibration-dot" style="bottom:5%; left:5%" onclick="calClick(this)"></div>
        <div id="cal4" class="calibration-dot" style="bottom:5%; right:5%" onclick="calClick(this)"></div>
        <div id="cal5" class="calibration-dot" style="top:50%; left:50%; transform:translate(-50%, -50%)" onclick="calClick(this)"></div>

        <script>
            let recordedData = [];
            let isRecording = false;

            window.onload = function() {{
                webgazer.setGazeListener(function(data, elapsedTime) {{
                    if (data == null) return;
                    if (isRecording) {{
                        // Simpan x, y, dan durasi dummy (1)
                        recordedData.push([Math.round(data.x), Math.round(data.y), 1]);
                    }}
                }}).begin();
                webgazer.showVideoPreview(false); 
                webgazer.showPredictionPoints(true); 
            }};

            function startCalibration() {{
                document.querySelectorAll('.calibration-dot').forEach(el => {{
                    el.style.display = 'block';
                    el.style.opacity = '0.7';
                    el.dataset.clicks = 0;
                }});
                document.getElementById('status').innerText = "Status: Klik 5x pada setiap titik merah.";
            }}

            function calClick(el) {{
                let clicks = parseInt(el.dataset.clicks || 0);
                clicks++;
                el.dataset.clicks = clicks;
                el.style.opacity = 1 - (clicks * 0.15); 
                
                if (clicks >= 5) {{
                    el.style.display = 'none';
                    checkCalibrationDone();
                }}
            }}

            function checkCalibrationDone() {{
                let remaining = 0;
                document.querySelectorAll('.calibration-dot').forEach(el => {{
                    if (el.style.display !== 'none') remaining++;
                }});
                
                if (remaining === 0) {{
                    document.getElementById('status').innerText = "Status: Kalibrasi Selesai. Siap Rekam.";
                    document.getElementById('recBtn').disabled = false;
                    webgazer.showPredictionPoints(false); 
                }}
            }}

            function startRecording() {{
                recordedData = [];
                isRecording = true;
                document.getElementById('status').innerText = "Status: SEDANG MEREKAM... Lihat Gambar!";
                document.getElementById('recBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
            }}

            function stopRecording() {{
                isRecording = false;
                document.getElementById('status').innerText = "Status: Selesai. Data disalin ke Clipboard!";
                document.getElementById('stopBtn').disabled = true;
                
                let jsonStr = JSON.stringify(recordedData);
                document.getElementById('outputData').value = jsonStr;
                document.getElementById('outputData').style.display = 'block'; // Tampilkan textarea
                
                navigator.clipboard.writeText(jsonStr).then(() => {{
                    alert("Data berhasil disalin! Silakan PASTE (Ctrl+V) di kolom Streamlit.");
                }}, () => {{
                    alert("Gagal copy otomatis. Silakan copy manual teks di kotak area.");
                }});
            }}
        </script>
    </body>
    </html>
    """

# ==========================================
# 2. LOGIKA HEATMAP & VISUALISASI
# ==========================================

DB_NAME = 'eyetracking_app.db'

def draw_display(dispsize, image_data=None):
    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    if image_data is not None:
        img = np.array(image_data)
        if img.shape[2] == 4: img = img[:, :, :3] # Remove alpha channel
        if img.dtype == 'uint8': img = img.astype('float32') / 255.0
        
        # Center image
        w, h = len(img[0]), len(img)
        x = int(dispsize[0]/2 - w/2)
        y = int(dispsize[1]/2 - h/2)
        
        # Safe slicing
        y_end, x_end = min(y + h, dispsize[1]), min(x + w, dispsize[0])
        h_crop, w_crop = y_end - y, x_end - x
        
        if h_crop > 0 and w_crop > 0:
            screen[y:y_end, x:x_end, :] += img[:h_crop, :w_crop, :]
    
    dpi = 100.0
    figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1]); ax.set_axis_off(); fig.add_axes(ax)
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)
    return fig, ax

def gaussian(x, sx, y=None, sy=None):
    if y == None: y = x
    if sy == None: sy = sx
    xo, yo = x/2, y/2
    M = np.zeros([y, x], dtype=float)
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy))))
    return M

def generate_heatmap_figure(gazepoints, dispsize, image_data=None, alpha=0.5, gaussianwh=200):
    fig, ax = draw_display(dispsize, image_data=image_data)
    gwh = gaussianwh
    gsdwh = gwh / 6
    gaus = gaussian(gwh, gsdwh)
    strt = int(gwh / 2)
    heatmapsize = (int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt))
    heatmap = np.zeros(heatmapsize, dtype=float)

    for i in range(len(gazepoints)):
        x = strt + int(gazepoints[i][0]) - int(gwh / 2)
        y = strt + int(gazepoints[i][1]) - int(gwh / 2)
        
        if (0 < x < dispsize[0]) and (0 < y < dispsize[1]):
             heatmap[y:y+gwh, x:x+gwh] += gaus * gazepoints[i][2]

    heatmap = heatmap[strt:dispsize[1]+strt, strt:dispsize[0]+strt]
    if np.any(heatmap > 0):
        lowbound = np.mean(heatmap[heatmap > 0])
        # PERBAIKAN DI SINI: np.NaN -> np.nan (lowercase)
        heatmap[heatmap < lowbound] = np.nan
        ax.imshow(heatmap, cmap='jet', alpha=alpha)
    ax.invert_yaxis()
    return fig

# ==========================================
# 3. DATABASE (DENGAN FIX MIGRATION)
# ==========================================

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    
    # Create table with all columns for fresh install
    c.execute('''CREATE TABLE IF NOT EXISTS tests 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  test_name TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                  gaze_json TEXT, 
                  image_path TEXT)''')
    
    # MIGRATION FIX: Cek apakah kolom 'gaze_json' sudah ada
    c.execute("PRAGMA table_info(tests)")
    columns = [info[1] for info in c.fetchall()]
    
    if 'gaze_json' not in columns:
        try:
            c.execute('ALTER TABLE tests ADD COLUMN gaze_json TEXT')
            print("Database Migrated: Added gaze_json column.")
        except Exception as e:
            print(f"Migration Error: {e}")
            
    conn.commit()
    conn.close()

def make_hashes(password): return hashlib.sha256(str.encode(password)).hexdigest()
def check_hashes(password, hashed): return make_hashes(password) == hashed

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =?', (username,))
    data = c.fetchall(); conn.close()
    return True if data and check_hashes(password, data[0][1]) else False

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username, password) VALUES (?,?)', (username, make_hashes(password)))
        conn.commit(); return True
    except: return False
    finally: conn.close()

def save_test_result(username, test_name, json_data, image_file_buffer):
    user_dir = f"data_storage/{username}"
    if not os.path.exists(user_dir): os.makedirs(user_dir)
    
    img_path = ""
    if image_file_buffer:
        img_path = f"{user_dir}/{test_name}_{int(time.time())}.png"
        with open(img_path, "wb") as f: f.write(image_file_buffer.getbuffer())
        
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('INSERT INTO tests(username, test_name, gaze_json, image_path) VALUES (?,?,?,?)', 
              (username, test_name, json_data, img_path))
    conn.commit(); conn.close()

# ==========================================
# 4. USER INTERFACE
# ==========================================

def main():
    st.set_page_config(page_title="EyeTrack Direct", layout="wide")
    init_db()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''

    # Sidebar
    st.sidebar.title("Navigasi")
    if st.session_state['logged_in']:
        menu = ["Dashboard", "Pengujian Langsung", "Analisis Heatmap", "Logout"]
        choice = st.sidebar.selectbox("Menu", menu)
        st.sidebar.success(f"User: {st.session_state['username']}")
    else:
        menu = ["Login", "Sign Up"]
        choice = st.sidebar.selectbox("Menu", menu)

    # --- HALAMAN LOGIN/SIGNUP ---
    if choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login_user(username, password):
                st.session_state['logged_in'] = True; st.session_state['username'] = username; st.rerun()
            else: st.error("Username atau Password salah")
    
    elif choice == "Sign Up":
        st.subheader("Buat Akun")
        new_u = st.text_input("Username Baru")
        new_p = st.text_input("Password Baru", type='password')
        if st.button("Sign Up"):
            if add_user(new_u, new_p): st.success("Akun berhasil dibuat. Silakan Login.")
            else: st.warning("Username sudah terpakai.")

    elif choice == "Logout":
        st.session_state['logged_in'] = False; st.rerun()

    # --- DASHBOARD ---
    elif choice == "Dashboard":
        st.title("Riwayat Pengujian")
        conn = sqlite3.connect(DB_NAME)
        # Ambil kolom yang ada saja untuk menghindari error jika db lama masih ada isu
        try:
            df = pd.read_sql_query(f"SELECT id, test_name, timestamp FROM tests WHERE username='{st.session_state['username']}'", conn)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error membaca database: {e}")
        conn.close()

    # --- PENGUJIAN LANGSUNG ---
    elif choice == "Pengujian Langsung":
        st.title("Setup Perekaman Langsung")
        
        col1 = st.columns(1)[0]
        
        # Konversi gambar untuk HTML
        bg_file = None
        img_b64 = None

        with col1:
            st.info("1. Upload gambar target.")
            bg_file = st.file_uploader("Gambar Target", type=['png', 'jpg', 'jpeg'])
            test_name = st.text_input("Nama Pengujian", "Tes_Webcam_1")
            
            if bg_file:
                # Konversi image ke base64 string agar bisa masuk ke HTML iframe
                bytes_data = bg_file.getvalue()
                img_b64 = base64.b64encode(bytes_data).decode()
                st.success("Gambar berhasil dimuat ke Tracker.")

            st.warning("PERHATIAN: Pastikan wajah terkena cahaya yang cukup.")
            
            # Panggil fungsi HTML dengan parameter gambar base64
            components.html(get_webgazer_html(img_b64), height=1080, scrolling=False)
            
            st.markdown("---")
            st.markdown("### Simpan Data")
            st.write("Paste data JSON di bawah ini setelah selesai merekam:")
            
            # Area paste data (JSON)
            gaze_json_input = st.text_area("Paste (Ctrl+V) Data JSON di sini:", height=150)
            
            if st.button("Simpan Hasil Pengujian"):
                if bg_file and gaze_json_input:
                    try:
                        # Validasi JSON
                        json_data = json.loads(gaze_json_input)
                        if isinstance(json_data, list) and len(json_data) > 0:
                            save_test_result(st.session_state['username'], test_name, gaze_json_input, bg_file)
                            st.success("Data Tersimpan! Silakan cek menu Analisis Heatmap.")
                        else:
                            st.error("Data kosong atau format salah.")
                    except ValueError:
                        st.error("Data yang di-paste bukan format JSON yang valid.")
                else:
                    st.error("Mohon upload gambar dan paste data tracking.")

    # --- ANALISIS HEATMAP ---
    elif choice == "Analisis Heatmap":
        st.title("Visualisasi Heatmap")
        conn = sqlite3.connect(DB_NAME)
        try:
            tests = pd.read_sql_query(f"SELECT id, test_name, timestamp, image_path, gaze_json FROM tests WHERE username='{st.session_state['username']}'", conn)
        except Exception:
             st.error("Database error atau belum ada data yang valid.")
             tests = pd.DataFrame()
        
        if not tests.empty:
            sel_test = st.selectbox("Pilih Data Pengujian", tests['test_name'].unique())
            row = tests[tests['test_name'] == sel_test].iloc[0]
            
            try:
                # Load JSON Data
                if row['gaze_json']:
                    gaze_data = json.loads(row['gaze_json'])
                else:
                    gaze_data = [] # Handle empty legacy data

                # Load Image
                if row['image_path'] and os.path.exists(row['image_path']):
                    img = Image.open(row['image_path'])
                    disp_size = img.size
                else:
                    st.warning("File gambar asli tidak ditemukan. Menggunakan ukuran default.")
                    img = None; disp_size = (1280, 720)

                # ==========================
                # SLIDER FIX LOGIC
                # ==========================
                total_points = len(gaze_data)
                chunk_size = 600 # asumsi ~10 detik (60Hz)
                num_chunks = int(np.ceil(total_points / chunk_size))
                
                t_idx = 0
                
                if num_chunks > 1:
                    st.info(f"Total data: {total_points} titik ({num_chunks} segmen waktu).")
                    t_idx = st.slider("Pilih Segmen Waktu (per ~10 detik)", 0, num_chunks - 1, 0)
                elif total_points > 0:
                    st.info(f"Data pendek ({total_points} titik). Menampilkan seluruh data.")
                    t_idx = 0
                else:
                    st.error("Data tracking kosong.")

                # Generate Heatmap Button
                if total_points > 0:
                    start = t_idx * chunk_size
                    end = min((t_idx + 1) * chunk_size, total_points)
                    subset = gaze_data[start:end]
                    
                    if st.button("Generate Heatmap"):
                        if len(subset) > 0:
                            with st.spinner("Memproses Heatmap..."):
                                fig = generate_heatmap_figure(subset, disp_size, image_data=img, alpha=0.6)
                                st.pyplot(fig, use_container_width=True)
                                st.write(f"Menampilkan data index {start} sampai {end}")
                        else:
                            st.warning("Segmen data ini kosong.")
            
            except Exception as e:
                st.error(f"Error memproses data: {e}")
        else:
            st.info("Belum ada data pengujian.")
        conn.close()

if __name__ == '__main__':
    main()