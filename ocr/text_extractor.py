import fitz  # PyMuPDF
import os
import sys
import contextlib
from config_file import PDF_PATHS

# STDERR 경고 제거용 context manager
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def extract_text_from_pdfs():
    for PDF_PATH in PDF_PATHS:
        if not os.path.exists(PDF_PATH):
            print(f"PDF 파일이 존재하지 않습니다: {PDF_PATH}")
            continue

        print(f"\n텍스트 추출 시작: {PDF_PATH}")
        base_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
        output_dir = os.path.dirname(PDF_PATH)
        OUTPUT_TXT = os.path.join(output_dir, f"{base_name}_text.txt")

        try:
            with suppress_stderr():
                doc = fitz.open(PDF_PATH)
        except Exception as e:
            print(f"  → PDF 열기 실패: {e}")
            continue

        try:
            with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text()
                    out.write(f"==== Page {page_num} ====\n")
                    out.write(text.strip() + "\n\n")
            print(f"  → 텍스트 저장 완료: {OUTPUT_TXT}")
        except Exception as e:
            print(f"  → 텍스트 추출 실패: {e}")

if __name__ == "__main__":
    extract_text_from_pdfs()
