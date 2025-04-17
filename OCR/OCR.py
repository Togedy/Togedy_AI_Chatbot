import camelot
import pandas as pd
import os
from OCR.config_file import PDF_PATHS

# 설정
PAGE_LIMIT = 10  # -1이면 전체 페이지 처리

for PDF_PATH in PDF_PATHS:
    if not os.path.exists(PDF_PATH):
        print(f"PDF 파일이 존재하지 않습니다: {PDF_PATH}")
        continue

    print(f"\n파일 처리 시작: {PDF_PATH}")

    base_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
    output_dir = os.path.dirname(PDF_PATH)
    OUTPUT_CSV = os.path.join(output_dir, f"{base_name}_tables.csv")
    OUTPUT_CLEANED_CSV = os.path.join(output_dir, f"{base_name}_tables_cleaned.csv")

    # Camelot으로 테이블 읽기
    page_range = '1-end' if PAGE_LIMIT == -1 else f'1-{PAGE_LIMIT}'
    tables = camelot.read_pdf(PDF_PATH, pages=page_range, flavor='lattice')  # 'lattice'는 라인 기반

    if tables.n == 0:
        print("테이블을 추출할 수 없습니다.")
        continue

    # 모든 테이블 합치기
    df_list = []
    for i, table in enumerate(tables):
        df = table.df
        df.insert(0, "Page", table.page)
        df_list.append(df)

    result_df = pd.concat(df_list, ignore_index=True)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"원본 테이블 저장 완료: {OUTPUT_CSV}")

    # 정제
    df_cleaned = result_df.dropna(how='all')

    if df_cleaned.columns.str.contains('Unnamed').any():
        df_cleaned.columns = df_cleaned.iloc[0]
        df_cleaned = df_cleaned[1:]

    df_cleaned = df_cleaned.apply(lambda col: col.map(lambda x: str(x).strip() if pd.notnull(x) else x))
    df_cleaned.reset_index(drop=True, inplace=True)

    df_cleaned.to_csv(OUTPUT_CLEANED_CSV, index=False, encoding="utf-8-sig")
    print(f"정제된 테이블 저장 완료: {OUTPUT_CLEANED_CSV}")
