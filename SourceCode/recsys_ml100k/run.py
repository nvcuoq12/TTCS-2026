"""
run.py — Khởi động CineRec Demo
Chạy: python run.py
Mở trình duyệt: http://localhost:8000
"""
import subprocess, sys, os, time, webbrowser, threading

def open_browser():
    time.sleep(3)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════╗
║      CINEREC — MovieLens ml-100k Demo                 ║
╠═══════════════════════════════════════════════════════╣
║  Dataset   : 943 users · 1682 phim · 100k ratings    ║
║  Mô hình   : User-CF · Item-CF · Matrix Factorization ║
║  Địa chỉ   : http://localhost:8000                    ║
║  Dừng      : Ctrl + C                                 ║
╚═══════════════════════════════════════════════════════╝
    """)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    threading.Thread(target=open_browser, daemon=True).start()
    subprocess.run([
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", "0.0.0.0", "--port", "8000", "--reload"
    ])
