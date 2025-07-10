import camelot
import pandas as pd
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

def extract_tables_from_pdfs(page_limit=-1):
    for PDF_PATH in PDF_PATHS:
        if not os.path.exists(PDF_PATH):
            print(f"PDF 파일이 존재하지 않습니다: {PDF_PATH}")
            continue

        print(f"\n파일 처리 시작: {PDF_PATH}")
        base_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
        output_dir = os.path.dirname(PDF_PATH)

        OUTPUT_CSV = os.path.join(output_dir, f"{base_name}_tables.csv")
        OUTPUT_CLEANED_CSV = os.path.join(output_dir, f"{base_name}_tables_cleaned.csv")

        page_range = '1-end' if page_limit == -1 else f'1-{page_limit}'

        # Camelot 테이블 추출 (lattice → stream fallback 포함)
        tables = None
        for flavor in ['lattice', 'stream']:
            with suppress_stderr():
                print(f"  → {flavor} 방식 시도 중...")
                try:
                    tables = camelot.read_pdf(PDF_PATH, pages=page_range, flavor=flavor)
                    if tables.n > 0:
                        print(f"    → 테이블 {tables.n}개 추출 성공 ({flavor})")
                        break
                except Exception as e:
                    print(f"    → {flavor} 실패: {e}")
                    continue

        if not tables or tables.n == 0:
            print("  → 테이블 추출 실패: lattice/stream 모두 실패")
            continue

        # 테이블 페이지 번호 추가 및 병합
        df_list = []
        for table in tables:
            df = table.df.copy()
            df.insert(0, "Page", table.page)
            df_list.append(df)

        df_all = pd.concat(df_list, ignore_index=True)
        df_all.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"  → 원본 테이블 저장 완료: {OUTPUT_CSV}")

        # 정제
        df_cleaned = df_all.dropna(how='all')
        if df_cleaned.columns.str.contains('Unnamed').any():
            df_cleaned.columns = df_cleaned.iloc[0]
            df_cleaned = df_cleaned[1:]

        df_cleaned = df_cleaned.apply(lambda col: col.map(lambda x: str(x).strip() if pd.notnull(x) else x))
        df_cleaned.reset_index(drop=True, inplace=True)

        df_cleaned.to_csv(OUTPUT_CLEANED_CSV, index=False, encoding="utf-8-sig")
        print(f"  → 정제된 테이블 저장 완료: {OUTPUT_CLEANED_CSV}")

if __name__ == "__main__":
    extract_tables_from_pdfs()
