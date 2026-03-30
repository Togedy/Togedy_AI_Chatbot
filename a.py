import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("모든 패키지 설치 완료")
    except subprocess.CalledProcessError:
        print("설치 중 오류 발생")

if __name__ == "__main__":
    install_requirements()