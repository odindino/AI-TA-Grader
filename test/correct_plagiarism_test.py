#!/usr/bin/env python3
"""
正確的抄襲檢測測試腳本 - 使用正確的方法名稱
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加項目根目錄到Python路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("🔍 抄襲檢測功能測試（正確版本）")
    print("=" * 50)
    
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
        print(f"📊 載入測試檔案: {test_file.name}")
        
        if not test_file.exists():
            print(f"❌ 檔案不存在: {test_file}")
            return
        
        df = pd.read_csv(test_file)
        print(f"✅ 載入成功: {len(df)} 筆資料, {len(df.columns)} 個欄位")
        
        # 顯示欄位
        print(f"📋 欄位: {list(df.columns)}")
        
        # 找出問題欄位 - 找包含"352902"或"352903"的欄位名稱（這些是實際的問題）
        exclude_cols = ['name', 'id', 'sis_id', 'section', 'section_id', 'section_sis_id', 
                       'submitted', 'attempt', '10.0', '10.0.1', 'n correct', 'n incorrect', 'score']
        question_cols = [col for col in df.columns 
                        if '352902' in col or '352903' in col or 
                        (col not in exclude_cols and len(col) > 50)]  # 長欄位名稱通常是問題
        print(f"📝 問題欄位數量: {len(question_cols)}")
        
        if not question_cols:
            print("❌ 找不到問題欄位")
            return
        
        # 測試第一個問題
        test_question = question_cols[0]
        print(f"\n🔍 測試問題: {test_question}")
        
        # 取得回答
        answers = df[test_question].fillna("").astype(str).tolist()
        print(f"📄 回答數量: {len(answers)}")
        
        # 顯示前幾個回答的前100字符
        for i, answer in enumerate(answers[:3]):
            print(f"   學生 {i}: {answer[:100]}...")
        
        # 使用正確的方法進行相似度檢測（本地方法）
        print(f"\n🔍 使用本地算法計算相似度...")
        result = similarity_detector.calculate_local_similarity(answers)
        
        if result and result['info']['status'] == 'success':
            similarity_matrix = result['matrix']
            flags = result['flags']
            
            print(f"✅ 相似度計算成功")
            print(f"📊 相似度矩陣形狀: {similarity_matrix.shape}")
            print(f"🚩 相似度標記: {flags}")
            
            # 顯示相似度矩陣的一部分
            print(f"\n📊 相似度矩陣 (前3x3):")
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
                print(f"   高相似度標記數量: {sum(1 for flag in flags if flag >= 1)}")
        else:
            print(f"❌ 相似度計算失敗: {result}")
            return
        
        # 測試多個問題
        print(f"\n📝 測試所有問題的抄襲檢測...")
        all_high_sim_counts = []
        
        for i, question in enumerate(question_cols[:3]):  # 只測試前3個問題
            print(f"   分析問題 {i+1}: {question[:50]}...")
            
            answers = df[question].fillna("").astype(str).tolist()
            result = similarity_detector.calculate_local_similarity(answers)
            
            if result and result['info']['status'] == 'success':
                similarity_matrix = result['matrix']
                
                # 計算高相似度對數量
                high_sim_count = 0
                for row_i in range(len(similarity_matrix)):
                    for col_j in range(row_i + 1, len(similarity_matrix[row_i])):
                        if similarity_matrix[row_i][col_j] > 0.7:
                            high_sim_count += 1
                
                all_high_sim_counts.append(high_sim_count)
                print(f"     高相似度對: {high_sim_count}")
            else:
                print(f"     計算失敗")
                all_high_sim_counts.append(0)
        
        print(f"\n📊 總結:")
        print(f"   測試問題數: {len(question_cols[:3])}")
        print(f"   總高相似度對: {sum(all_high_sim_counts)}")
        print(f"   平均每題高相似度對: {np.mean(all_high_sim_counts):.1f}")
        
        print(f"\n🎉 抄襲檢測測試完成！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
