import ctypes

# 경로 바꾸지 말고 그대로 실행해 보세요
try:
    ctypes.WinDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cudnn_ops_infer64_8.dll")
    print("✅ DLL 로딩 성공")
except Exception as e:
    print("❌ 로딩 실패:", e)
