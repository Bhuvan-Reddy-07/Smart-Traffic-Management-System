import sys
import subprocess
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
APP_PATH = ROOT / "Smart_Traffic_Management_System" / "Smart_Traffic_Management_System" / "app.py"

if __name__ == "__main__":
    print(f"Launching Smart Traffic Management System from: {APP_PATH}")
    subprocess.run([sys.executable, str(APP_PATH)], cwd=str(APP_PATH.parent))
