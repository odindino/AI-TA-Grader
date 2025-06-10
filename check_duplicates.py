#!/usr/bin/env python3
# check_duplicates.py
# 檢查 CSV 檔案中的重複學生記錄

import pandas as pd
import sys
import os

def analyze_duplicates(csv_path):
    """分析 CSV 檔案中的重複記錄"""
    print(f"=== 分析重複記錄: {os.path.basename(csv_path)} ===")
    
    try:
        # 載入 CSV
        df = pd.read_csv(
            csv_path,
            encoding='utf-8',
            dtype=str,
            quotechar='"',
            escapechar='\\'
        )
        
        # 清理欄位名稱
        df.columns = df.columns.str.strip().str.replace('\n', ' ', regex=False)
        
        print(f"📊 總行數: {len(df)}")
        
        # 檢查 name 欄位
        if 'name' not in df.columns:
            print("❌ 找不到 'name' 欄位")
            print("可用欄位:", list(df.columns[:10]))
            return
        
        # 顯示所有學生姓名
        print(f"\n📝 所有學生記錄 ({len(df)} 行):")
        for i, name in enumerate(df['name'], 1):
            print(f"  {i:2d}. {name}")
        
        # 檢查重複
        name_counts = df['name'].value_counts()
        duplicates = name_counts[name_counts > 1]
        
        print(f"\n🔍 重複分析:")
        print(f"  唯一學生數: {len(name_counts)}")
        print(f"  重複學生數: {len(duplicates)}")
        
        if len(duplicates) > 0:
            print(f"\n⚠️  重複的學生:")
            for name, count in duplicates.items():
                print(f"  - {name}: {count} 次")
                
                # 顯示重複記錄的詳細資訊
                duplicate_rows = df[df['name'] == name]
                print(f"    行號: {list(duplicate_rows.index + 1)}")
                
                # 檢查這些重複記錄是否完全相同
                if len(duplicate_rows.drop_duplicates()) == 1:
                    print("    → 完全相同的記錄")
                else:
                    print("    → 記錄內容有差異")
                    # 顯示差異的欄位
                    for col in df.columns:
                        unique_values = duplicate_rows[col].dropna().unique()
                        if len(unique_values) > 1:
                            print(f"      {col}: {list(unique_values)}")
                print()
        else:
            print("✅ 沒有發現重複的學生")
        
        # 檢查空白姓名
        empty_names = df['name'].isna() | (df['name'].str.strip() == '')
        empty_count = empty_names.sum()
        
        if empty_count > 0:
            print(f"⚠️  空白姓名記錄: {empty_count} 行")
            empty_indices = df[empty_names].index + 1
            print(f"   行號: {list(empty_indices)}")
        
        # 顯示去重後的結果
        df_cleaned = df.drop_duplicates(subset=['name'], keep='first')
        df_cleaned = df_cleaned.dropna(subset=['name'])
        df_cleaned = df_cleaned[df_cleaned['name'].str.strip() != '']
        
        print(f"\n✨ 清理後結果:")
        print(f"  最終學生數: {len(df_cleaned)}")
        print(f"  清理過程: {len(df)} → {len(df_cleaned)} 行")
        
    except Exception as e:
        print(f"❌ 分析過程發生錯誤: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "testfile/Final Exam Quiz Student Analysis Report.csv"
    
    analyze_duplicates(csv_file)
