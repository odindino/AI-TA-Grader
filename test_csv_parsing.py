#!/usr/bin/env python3
# test_csv_parsing.py
# 測試CSV結構解析功能

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import load_exam

def test_csv_parsing(test_file=None):
    """測試CSV解析功能"""
    print("=== CSV結構解析測試 ===")
    print(f"📁 測試檔案: {test_file}")
    print("-" * 50)
    
    # 模擬log回調函數
    def log_callback(message):
        print(f"LOG: {message}")
    
    # 載入並解析
    df, q_map = load_exam(test_file, log_callback)
    
    if df is not None and q_map is not None:
        print(f"\n📊 解析結果:")
        print(f"   學生數量: {len(df)}")
        print(f"   題目數量: {len(q_map)}")
        
        print(f"\n📝 學生名單:")
        for i, name in enumerate(df['name'].head(5), 1):
            print(f"   {i}. {name}")
        
        print(f"\n❓ 題目列表:")
        for qid, question_col in q_map.items():
            # 截取題目前80字符顯示
            question_preview = question_col[:80] + "..." if len(question_col) > 80 else question_col
            print(f"   Q{qid}: {question_preview}")
        
        print(f"\n🔍 第一位學生的回答範例:")
        first_student = df.iloc[0]
        print(f"   學生: {first_student['name']}")
        for qid, question_col in list(q_map.items())[:2]:  # 只顯示前兩題
            answer = first_student[question_col]
            answer_preview = str(answer)[:120] + "..." if len(str(answer)) > 120 else str(answer)
            print(f"   Q{qid} 回答: {answer_preview}")
    else:
        print("❌ 解析失敗")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = "testfile/structure_test.csv"
    
    test_csv_parsing(test_file)
