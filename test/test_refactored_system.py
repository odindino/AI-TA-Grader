#!/usr/bin/env python3
"""
完整重構系統測試
測試所有新的backend模組和整合功能
"""

import sys
import os
import asyncio

# 添加路徑
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'backend'))

async def test_backend_modules():
    """測試backend模組功能"""
    print("=== 測試Backend模組 ===")
    
    try:
        # 測試配置模組
        from backend.config import TARGET_SCORE_Q, RUBRICS, SIMILARITY_THRESHOLDS
        print(f"✅ Config模組載入成功")
        print(f"   - 目標問題: {TARGET_SCORE_Q}")
        print(f"   - 相似度閾值: {SIMILARITY_THRESHOLDS}")
        
        # 測試數據處理模組
        from backend.data_processor import DataProcessor
        processor = DataProcessor()
        print(f"✅ DataProcessor模組載入成功")
        
        # 測試視覺化模組
        from backend.visualization import VisualizationEngine
        viz = VisualizationEngine()
        print(f"✅ VisualizationEngine模組載入成功")
        
        # 測試相似度檢測模組
        from backend.similarity_detector import SimilarityDetector
        detector = SimilarityDetector()
        print(f"✅ SimilarityDetector模組載入成功")
        
        # 測試Gemini客戶端模組（無API key）
        from backend.gemini_client import GeminiClient
        print(f"✅ GeminiClient模組載入成功")
        
        # 測試主分析引擎
        from backend.analyzer import AnalysisEngine
        engine = AnalysisEngine()
        print(f"✅ AnalysisEngine模組載入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend模組測試失敗: {e}")
        return False

async def test_analysis_engine():
    """測試分析引擎功能"""
    print("\n=== 測試分析引擎功能 ===")
    
    try:
        from backend.analyzer import AnalysisEngine
        
        # 初始化引擎（無API key）
        engine = AnalysisEngine()
        print("✅ 分析引擎初始化成功")
        
        # 測試CSV檔案路徑
        test_csv = os.path.join(parent_dir, 'testfile', 'Final Exam Quiz Student Analysis Report_Public.csv')
        
        if not os.path.exists(test_csv):
            print(f"❌ 測試檔案不存在: {test_csv}")
            return False
        
        # 日誌回調
        def log_callback(message):
            print(f"   📋 {message}")
        
        # 執行完整分析
        print("🚀 開始完整數據集分析...")
        results = await engine.analyze_complete_dataset(test_csv, log_callback)
        
        # 驗證結果
        if 'dataframe' in results:
            df = results['dataframe']
            print(f"✅ 數據框創建成功，包含 {len(df)} 行")
            
            # 檢查是否有分數欄位
            score_columns = [col for col in df.columns if '_分數' in col]
            print(f"✅ 找到 {len(score_columns)} 個分數欄位: {score_columns}")
            
        if 'html_report' in results:
            print(f"✅ HTML報告生成成功，長度: {len(results['html_report'])} 字符")
            
        if 'summary' in results:
            summary = results['summary']
            print(f"✅ 分析摘要生成成功")
            print(f"   - 學生總數: {summary.get('total_students', 0)}")
            print(f"   - 分析問題: {summary.get('questions_analyzed', [])}")
            
        return True
        
    except Exception as e:
        print(f"❌ 分析引擎測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_similarity_detection():
    """測試相似度檢測功能"""
    print("\n=== 測試相似度檢測 ===")
    
    try:
        from backend.similarity_detector import SimilarityDetector
        
        detector = SimilarityDetector()
        
        # 測試文本
        test_texts = [
            "This is a test answer about semiconductors.",
            "This is a test answer about semiconductors with small changes.",
            "Completely different answer about biology.",
            "Another unique response about chemistry."
        ]
        
        # 測試本地相似度檢測
        print("🔍 測試本地相似度檢測...")
        results = detector.calculate_local_similarity(test_texts)
        
        print(f"✅ 相似度標記: {results['flags']}")
        print(f"✅ 檢測信息: {results['info']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 相似度檢測測試失敗: {e}")
        return False

async def test_visualization():
    """測試視覺化功能"""
    print("\n=== 測試視覺化功能 ===")
    
    try:
        from backend.visualization import VisualizationEngine
        
        viz = VisualizationEngine()
        
        # 測試數據
        test_texts = [
            "Answer 1 about topic A",
            "Answer 2 about topic B", 
            "Answer 3 about topic C"
        ]
        test_names = ["Student1", "Student2", "Student3"]
        
        # 測試相似度矩陣創建
        print("📊 測試相似度矩陣創建...")
        matrices = await viz.create_similarity_matrices(
            test_texts, test_names, question_id=1, use_genai=False
        )
        
        if matrices.get('local_matrix'):
            print("✅ 本地相似度矩陣創建成功")
        
        # 測試HTML報告生成
        print("🌐 測試HTML報告生成...")
        import pandas as pd
        test_df = pd.DataFrame({
            'Student': test_names,
            'Q1_分數': [8.5, 7.2, 9.1],
            'Q1_相似度標記': [0, 0, 0]
        })
        
        html_report = await viz.generate_enhanced_html_report(
            test_df, {'Q1': matrices}
        )
        
        if html_report and len(html_report) > 100:
            print("✅ HTML報告生成成功")
        else:
            print("⚠️ HTML報告可能有問題")
            
        return True
        
    except Exception as e:
        print(f"❌ 視覺化測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主測試函數"""
    print("🧪 AI-TA-Grader 重構系統完整測試")
    print("=" * 60)
    
    # 執行所有測試
    tests = [
        ("Backend模組", test_backend_modules),
        ("相似度檢測", test_similarity_detection),
        ("視覺化功能", test_visualization),
        ("分析引擎", test_analysis_engine),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 執行測試: {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} 測試通過")
            else:
                print(f"❌ {test_name} 測試失敗")
        except Exception as e:
            print(f"❌ {test_name} 測試異常: {e}")
    
    print(f"\n📊 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！重構系統運行正常。")
    else:
        print("⚠️ 部分測試失敗，請檢查錯誤信息。")

if __name__ == "__main__":
    asyncio.run(main())
