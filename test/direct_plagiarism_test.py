#!/usr/bin/env python3
"""
直接測試抄襲檢測功能
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加項目根目錄到Python路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("🔍 直接測試抄襲檢測功能")
    print("=" * 40)
    
    try:
        # 匯入模組
        from backend.data_processor import DataProcessor
        from backend.similarity_detector import SimilarityDetector
        print("✅ 模組匯入成功")
        
        # 初始化組件
        data_processor = DataProcessor()
        similarity_detector = SimilarityDetector()
        print("✅ 組件初始化成功")
        
        # 載入測試資料
        test_file = project_root / "testfile" / "Final Exam Quiz Student Analysis Report_Public_plag.csv"
        print(f"📊 載入測試檔案: {test_file}")
        
        if not test_file.exists():
            print(f"❌ 檔案不存在: {test_file}")
            return
        
        df = pd.read_csv(test_file)
        print(f"✅ 載入成功: {len(df)} 筆資料, {len(df.columns)} 個欄位")
        
        # 顯示欄位
        print(f"📋 欄位: {list(df.columns)}")
        
        # 找出問題欄位
        exclude_cols = ['Student', 'Email', 'Timestamp']
        question_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"📝 問題欄位: {question_cols}")
        
        if not question_cols:
            print("❌ 找不到問題欄位")
            return
        
        # 測試第一個問題
        test_question = question_cols[0]
        print(f"\n🔍 測試問題: {test_question}")
        
        # 取得回答
        answers = df[test_question].fillna("").astype(str).tolist()
        print(f"📄 回答數量: {len(answers)}")
        
        # 顯示前幾個回答
        for i, answer in enumerate(answers[:3]):
            print(f"   學生 {i}: {answer[:100]}...")
        
        # 計算相似度矩陣
        print("\n🔍 計算相似度矩陣...")
        similarity_matrix = similarity_detector.calculate_similarity_matrix(answers)
        print(f"✅ 相似度矩陣計算完成: {len(similarity_matrix)}x{len(similarity_matrix[0])}")
        
        # 顯示相似度矩陣的一部分
        print("\n📊 相似度矩陣 (前3x3):")
        for i in range(min(3, len(similarity_matrix))):
            row = similarity_matrix[i]
            row_str = [f"{val:.3f}" for val in row[:3]]
            print(f"   [{', '.join(row_str)}]")
        
        # 找出高相似度對
        print(f"\n🚨 尋找高相似度對 (閾值: 70%)...")
        high_sim_pairs = []
        
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix[i])):
                similarity = similarity_matrix[i][j]
                if similarity > 0.7:
                    high_sim_pairs.append((i, j, similarity))
        
        # 按相似度排序
        high_sim_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"✅ 發現 {len(high_sim_pairs)} 對高相似度回答:")
        for i, (student1, student2, sim) in enumerate(high_sim_pairs[:5]):
            print(f"   {i+1}. 學生 {student1} vs 學生 {student2}: {sim:.3f}")
            
            # 顯示回答內容
            answer1 = answers[student1][:100]
            answer2 = answers[student2][:100]
            print(f"      學生 {student1}: {answer1}...")
            print(f"      學生 {student2}: {answer2}...")
            print()
        
        # 統計資訊
        all_similarities = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix[i])):
                all_similarities.append(similarity_matrix[i][j])
        
        if all_similarities:
            avg_sim = np.mean(all_similarities)
            max_sim = np.max(all_similarities)
            print(f"📊 統計資訊:")
            print(f"   平均相似度: {avg_sim:.3f}")
            print(f"   最高相似度: {max_sim:.3f}")
            print(f"   相似度對數量: {len(all_similarities)}")
        
        print("\n🎉 測試完成！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
