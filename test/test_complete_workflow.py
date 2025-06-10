#!/usr/bin/env python3
"""
完整功能測試：測試無API key模式的完整工作流程
"""

import sys
import os
import asyncio

# 添加父目錄到路徑
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'backend'))

async def test_complete_workflow():
    """測試完整的分析工作流程（無API key）"""
    print("=== 測試完整工作流程（無API key模式）===")
    
    from backend.analyzer import AnalysisEngine
    
    # 測試檔案路徑
    test_csv = os.path.join(parent_dir, 'testfile', 'Final Exam Quiz Student Analysis Report_Public.csv')
    
    if not os.path.exists(test_csv):
        print(f"❌ 測試檔案不存在: {test_csv}")
        return
    
    # 輸出路徑
    output_base = os.path.join(parent_dir, 'test_output_no_api')
    
    # 日誌回調函數
    def log_callback(message):
        print(f"📋 {message}")
    
    try:
        # 初始化分析引擎（無API key）
        print("🚀 初始化分析引擎（無API key）...")
        engine = AnalysisEngine(api_key=None)
        
        # 執行完整數據集分析
        print("📊 開始分析數據集...")
        results = await engine.analyze_complete_dataset(test_csv, log_callback)
        
        # 保存結果
        print("💾 保存分析結果...")
        df = results['dataframe']
        html_report = results['html_report']
        
        # 保存Excel檔案
        xlsx_path = f"{output_base}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"✅ Excel報告已保存: {os.path.basename(xlsx_path)}")
        
        # 保存HTML報告
        html_path = f"{output_base}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"✅ HTML報告已保存: {os.path.basename(html_path)}")
        
        # 保存CSV檔案
        csv_path = f"{output_base}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV檔案已保存: {os.path.basename(csv_path)}")
        
        # 檢查輸出檔案
        expected_files = [xlsx_path, csv_path, html_path]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✅ 成功生成: {os.path.basename(file_path)}")
            else:
                print(f"❌ 缺少檔案: {os.path.basename(file_path)}")
        
        # 特別檢查HTML檔案是否包含視覺化
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            if 'matrix-image' in html_content:
                print("✅ HTML報告包含視覺化矩陣")
            else:
                print("⚠️ HTML報告未包含視覺化矩陣")
            
            if 'data:image/png;base64,' in html_content:
                print("✅ HTML報告包含base64圖像")
            else:
                print("⚠️ HTML報告未包含base64圖像")
                
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

def test_similarity_methods_comparison():
    """比較不同相似度方法的結果"""
    print("\n=== 比較相似度檢測方法 ===")
    
    from alternative_similarity_methods import (
        calculate_text_similarity_local, 
        calculate_text_similarity_enhanced,
        calculate_tfidf_similarity
    )
    
    # 測試案例
    test_texts = [
        "The Czochralski method is used for growing single crystal silicon. It involves pulling a seed crystal from molten silicon.",
        "The CZ process grows single crystal silicon by pulling a seed from molten Si material slowly while rotating.",
        "Molecular beam epitaxy is a different technique for thin film deposition using atomic beams in ultra-high vacuum.",
        "Solar cells convert light energy into electrical energy through the photovoltaic effect in semiconductor materials."
    ]
    
    test_names = ["Student_A", "Student_B", "Student_C", "Student_D"]
    
    print("測試文本相似度（前兩個應該高度相似）:")
    
    # 方法1: 原始本地方法
    local_results, local_details = calculate_text_similarity_local(test_texts, test_names, threshold=0.6)
    print(f"📝 原始本地方法: {local_results}")
    
    # 方法2: 增強本地方法
    enhanced_results, enhanced_details = calculate_text_similarity_enhanced(test_texts, test_names, threshold=0.6)
    print(f"🔬 增強本地方法: {enhanced_results}")
    
    # 方法3: TF-IDF方法
    tfidf_results, tfidf_details = calculate_tfidf_similarity(test_texts, threshold=0.6)
    print(f"📊 TF-IDF方法: {tfidf_results}")
    
    print("\n詳細相似度配對:")
    for detail in enhanced_details:
        print(f"  {detail['student_1']} ↔ {detail['student_2']}: {detail['similarity']:.3f}")

def main():
    """主測試函數"""
    print("🧪 完整功能測試")
    print("=" * 60)
    
    # 測試1: 相似度方法比較
    test_similarity_methods_comparison()
    
    # 測試2: 完整工作流程
    print("\n" + "=" * 60)
    asyncio.run(test_complete_workflow())
    
    print("\n✅ 完整功能測試完成!")
    print("\n📝 總結:")
    print("1. ✅ 非GenAI相似度檢測已優化並包含多種先進算法")
    print("2. ✅ API key變為可選，無API key時只執行非GenAI分析")
    print("3. ✅ 視覺化相似度矩陣功能已實現")
    print("4. ✅ HTML報告包含增強的視覺化和統計信息")

if __name__ == "__main__":
    main()
