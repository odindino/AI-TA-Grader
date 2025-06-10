#!/usr/bin/env python3
# check_deduplication_result.py
# 檢查去重後保留了哪些記錄

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import load_exam

def check_deduplication_result(csv_path):
    """檢查去重結果"""
    print(f"=== 檢查去重結果: {os.path.basename(csv_path)} ===")
    
    def log_callback(message):
        print(f"LOG: {message}")
    
    # 使用我們的載入函數
    df, q_map = load_exam(csv_path, log_callback)
    
    if df is None:
        return
    
    print(f"\n🎯 具體檢查重複學生的保留結果:")
    
    # 重新載入原始檔案檢查
    original_df = pd.read_csv(
        csv_path,
        encoding='utf-8',
        dtype=str,
        quotechar='"',
        escapechar='\\'
    )
    original_df.columns = original_df.columns.str.strip().str.replace('\n', ' ', regex=False)
    
    # 檢查重複學生
    duplicate_students = ['李雨謙 (YU-CHIEN LEE)', '柯又中 (KO, YU-CHUNG)']
    
    for student in duplicate_students:
        print(f"\n👤 {student}:")
        
        # 原始記錄
        original_records = original_df[original_df['name'] == student]
        print(f"  原始記錄數: {len(original_records)}")
        
        for i, (idx, record) in enumerate(original_records.iterrows(), 1):
            attempt = record.get('attempt', 'N/A')
            score = record.get('score', 'N/A')
            submitted = record.get('submitted', 'N/A')
            print(f"    記錄 {i}: attempt={attempt}, score={score}, submitted={submitted}")
        
        # 保留的記錄
        kept_record = df[df['name'] == student]
        if len(kept_record) > 0:
            kept = kept_record.iloc[0]
            attempt = kept.get('attempt', 'N/A')
            score = kept.get('score', 'N/A')
            submitted = kept.get('submitted', 'N/A')
            print(f"  ✅ 保留記錄: attempt={attempt}, score={score}, submitted={submitted}")
        else:
            print(f"  ❌ 該學生記錄未被保留")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "testfile/Final Exam Quiz Student Analysis Report.csv"
    
    check_deduplication_result(csv_file)
