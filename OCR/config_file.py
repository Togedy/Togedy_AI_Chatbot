import os

# 절대 경로로 BASE_PATH 설정
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../university"))

UNIVERSITY_MAP = {
    "hanyang": ["susi", "jungsi"],
    "konkuk": ["susi", "jungsi"],
    "korea": ["susi", "jungsi"],
    "seoul": ["susi", "jungsi"],
    "skku": ["susi", "jungsi"],
    "sogang": ["susi", "jungsi"],
    "yonsei": ["susi", "jungsi"]
}

PDF_PATHS = [
    os.path.join(BASE_PATH, university, f"{exam_type}.pdf")
    for university, exam_types in UNIVERSITY_MAP.items()
    for exam_type in exam_types
]
