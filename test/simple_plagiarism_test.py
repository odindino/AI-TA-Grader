#!/usr/bin/env python3
"""
簡化的抄襲檢測測試腳本 - 無API模式
專注於核心功能測試
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加項目根目錄到Python路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"📁 項目根目錄: {project_root}")
print(f"🐍 Python路徑: {sys.path[:3]}")

def test_backend_imports():
    """測試後端模組匯入"""
    print("\n🔧 測試後端模組匯入...")
    
    try:
        from backend.data_processor import DataProcessor
        print("✅ DataProcessor 匯入成功")
        
        from backend.similarity_detector import SimilarityDetector
        print("✅ SimilarityDetector 匯入成功")
        
        from backend.visualization import VisualizationEngine
        print("✅ VisualizationEngine 匯入成功")
        
        from backend.config import Config
        print("✅ Config 匯入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 模組匯入失敗: {e}")
        return False

def test_plagiarism_detection():
    """測試抄襲檢測核心功能"""
    print("\n🔍 測試抄襲檢測功能...")
    
    try:
        from backend.data_processor import DataProcessor
        from backend.similarity_detector import SimilarityDetector
        
        # 初始化組件
        data_processor = DataProcessor()
        similarity_detector = SimilarityDetector()
        
        # 載入測試資料
        test_file = project_root / "testfile" / "Final Exam Quiz Student Analysis Report_Public_plag.csv"
        
        if not test_file.exists():
            print(f"❌ 測試檔案不存在: {test_file}")
            return False
        
        print(f"📊 載入測試檔案: {test_file.name}")
        df = pd.read_csv(test_file)
        print(f"✅ 成功載入 {len(df)} 筆資料，{len(df.columns)} 個欄位")
        
        # 找出問題欄位
        exclude_cols = ['Student', 'Email', 'Timestamp']
        question_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"📝 找到 {len(question_cols)} 個問題欄位")
        
        # 測試第一個問題的相似度檢測
        if question_cols:
            first_question = question_cols[0]
            print(f"\n🔍 分析問題: {first_question}")
            
            # 取得回答
            answers = df[first_question].fillna("").astype(str).tolist()
            print(f"📄 回答數量: {len(answers)}")
            
            # 檢測相似度
            similarity_matrix = similarity_detector.calculate_similarity_matrix(answers)
            print(f"📊 相似度矩陣形狀: {np.array(similarity_matrix).shape}")
            
            # 找出高相似度對
            high_similarity_pairs = []
            threshold = 0.7
            
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix[i])):
                    similarity = similarity_matrix[i][j]
                    if similarity > threshold:
                        high_similarity_pairs.append((i, j, similarity))
            
            print(f"\n🚨 發現 {len(high_similarity_pairs)} 對高相似度回答 (>70%):")
            for i, j, sim in high_similarity_pairs[:5]:  # 顯示前5個
                print(f"   學生 {i} vs 學生 {j}: {sim:.3f}")
                print(f"   回答 {i}: {answers[i][:100]}...")
                print(f"   回答 {j}: {answers[j][:100]}...")
                print()
            
            return True
        else:
            print("❌ 找不到問題欄位")
            return False
            
    except Exception as e:
        print(f"❌ 抄襲檢測測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization():
    """測試視覺化功能"""
    print("\n🎨 測試視覺化功能...")
    
    try:
        from backend.visualization import VisualizationEngine
        
        viz_engine = VisualizationEngine()
        
        # 創建測試資料
        test_matrix = [
            [1.0, 0.3, 0.9],  # 學生0與學生2高度相似
            [0.3, 1.0, 0.4],
            [0.9, 0.4, 1.0]
        ]
        
        # 測試相似度矩陣視覺化
        result = viz_engine.create_similarity_matrices({"測試問題": test_matrix})
        
        if result:
            print("✅ 視覺化生成成功")
            print(f"📏 結果長度: {len(result)} 字符")
            return True
        else:
            print("❌ 視覺化生成失敗")
            return False
            
    except Exception as e:
        print(f"❌ 視覺化測試失敗: {e}")
        return False

def generate_simple_report():
    """生成簡單的測試報告"""
    print("\n📄 生成測試報告...")
    
    try:
        from backend.data_processor import DataProcessor
        from backend.similarity_detector import SimilarityDetector
        from backend.visualization import VisualizationEngine
        
        # 載入資料
        test_file = project_root / "testfile" / "Final Exam Quiz Student Analysis Report_Public_plag.csv"
        df = pd.read_csv(test_file)
        
        # 初始化組件
        data_processor = DataProcessor()
        similarity_detector = SimilarityDetector()
        viz_engine = VisualizationEngine()
        
        # 分析所有問題
        exclude_cols = ['Student', 'Email', 'Timestamp']
        question_cols = [col for col in df.columns if col not in exclude_cols]
        
        all_similarities = {}
        report_content = []
        
        report_content.append("<html><head><title>抄襲檢測報告</title></head><body>")
        report_content.append("<h1>AI-TA-Grader 抄襲檢測報告</h1>")
        report_content.append(f"<p>測試時間: {pd.Timestamp.now()}</p>")
        report_content.append(f"<p>測試模式: 離線模式（無API）</p>")
        report_content.append(f"<p>學生數量: {len(df)}</p>")
        report_content.append(f"<p>問題數量: {len(question_cols)}</p>")
        
        for i, question in enumerate(question_cols):
            print(f"🔍 分析問題 {i+1}/{len(question_cols)}: {question}")
            
            answers = df[question].fillna("").astype(str).tolist()
            similarity_matrix = similarity_detector.calculate_similarity_matrix(answers)
            all_similarities[question] = similarity_matrix
            
            # 統計高相似度對
            high_sim_count = 0
            max_similarity = 0
            
            for row_i in range(len(similarity_matrix)):
                for col_j in range(row_i + 1, len(similarity_matrix[row_i])):
                    sim = similarity_matrix[row_i][col_j]
                    max_similarity = max(max_similarity, sim)
                    if sim > 0.7:
                        high_sim_count += 1
            
            report_content.append(f"<h2>問題 {i+1}: {question}</h2>")
            report_content.append(f"<p>最高相似度: {max_similarity:.3f}</p>")
            report_content.append(f"<p>高相似度對數 (>70%): {high_sim_count}</p>")
        
        report_content.append("</body></html>")
        
        # 儲存報告
        output_dir = project_root / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / "simple_plagiarism_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"✅ 報告已儲存: {report_path}")
        return True
        
    except Exception as e:
        print(f"❌ 報告生成失敗: {e}")
        return False

def main():
    """主函數"""
    print("🔍 AI-TA-Grader 簡化抄襲檢測測試")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # 測試1: 模組匯入
    if test_backend_imports():
        success_count += 1
    
    # 測試2: 抄襲檢測
    if test_plagiarism_detection():
        success_count += 1
    
    # 測試3: 視覺化
    if test_visualization():
        success_count += 1
    
    # 測試4: 報告生成
    if generate_simple_report():
        success_count += 1
    
    print(f"\n📊 測試結果: {success_count}/{total_tests} 通過")
    
    if success_count == total_tests:
        print("🎉 所有測試通過！")
    else:
        print("⚠️ 部分測試失敗")
    
    return success_count == total_tests

if __name__ == "__main__":
    main()
