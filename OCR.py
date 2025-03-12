import os
import pdfplumber
import pandas as pd
from config_file import PDF_PATH

# PDF 파일 경로 설정
pdf_path = PDF_PATH
output_text_csv = os.path.splitext(pdf_path)[0] + "_text.csv"
output_table_csv = os.path.splitext(pdf_path)[0] + "_table.csv"

# 결과 저장을 위한 리스트
text_data = []
table_data = []

# PDF에서 텍스트 및 표 추출
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        # 텍스트 데이터 추출 및 저장
        text = page.extract_text()
        if text:
            text_data.append(["Page", i + 1, text.strip()])

        # 표 데이터 추출 및 병합된 셀 보정
        tables = page.extract_tables()
        for table in tables:
            max_columns = max(len(row) for row in table if row)  # 가장 긴 행의 열 개수
            
            for row in table:
                if not row:
                    continue  # 빈 행은 스킵
                
                # 빈 칸을 이전 행의 값으로 채움 (병합된 셀 보정)
                cleaned_row = []
                for col_index, col in enumerate(row):
                    if col and col.strip():
                        cleaned_row.append(col.strip())
                    elif len(table_data) > 0 and col_index < len(table_data[-1]):
                        cleaned_row.append(table_data[-1][col_index])  # 이전 행의 값을 채움
                    else:
                        cleaned_row.append("없음")  # 완전히 비어 있는 경우만 '없음' 처리
                
                # 열 개수가 부족하면 '없음'으로 채우기
                while len(cleaned_row) < max_columns:
                    cleaned_row.append("없음")

                table_data.append([i + 1] + cleaned_row)  # 페이지 번호 추가

# 텍스트 CSV 저장
df_text = pd.DataFrame(text_data, columns=["Type", "Page", "Content"])
df_text.to_csv(output_text_csv, index=False, encoding="utf-8-sig")

# 표 CSV 저장
df_table = pd.DataFrame(table_data)
df_table.to_csv(output_table_csv, index=False, encoding="utf-8-sig")

print(f"텍스트 데이터가 {output_text_csv} 파일에 저장되었습니다.")
print(f"표 데이터가 {output_table_csv} 파일에 저장되었습니다.")
