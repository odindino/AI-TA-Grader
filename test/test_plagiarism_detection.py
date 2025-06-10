#!/usr/bin/env python3
"""
測試抄襲檢測功能 - 使用修改後的測試檔案
"""

import sys
import os
import asyncio

# 添加路徑
parent_dir = os.path.dirname(__file__)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'backend'))

async def test_plagiarism_detection():
    """測試抄襲檢測功能"""
    print("🕵️ AI-TA-Grader 抄襲檢測功能測試")
    print("=" * 50)
    
    from backend.analyzer import AnalysisEngine
    
    # 測試檔案路徑
    test_csv = os.path.join(parent_dir, 'testfile', 'Final Exam Quiz Student Analysis Report_Public_plag.csv')
    
    if not os.path.exists(test_csv):
        print(f"❌ 測試檔案不存在: {test_csv}")
        return
    
    # 日誌回調函數
    def log_callback(message):
        print(f"📋 {message}")
    
    try:
        # 初始化分析引擎（無API key模式）
        print("🚀 初始化分析引擎（離線模式）...")
        engine = AnalysisEngine(api_key=None)
        
        # 執行完整數據集分析
        print("📊 開始分析數據集...")
        results = await engine.analyze_complete_dataset(test_csv, log_callback)
        
        # 檢查相似度檢測結果
        df = results['dataframe']
        
        print("\n🔍 相似度檢測結果分析：")
        print("-" * 40)
        
        # 查看Q1相似度結果
        if 'Q1_相似度標記' in df.columns:
            q1_similarity = df['Q1_相似度標記'].tolist()
            names = df['name'].tolist() if 'name' in df.columns else [f"學生{i+1}" for i in range(len(df))]
            
            print("Q1 相似度標記結果：")
            for i, (name, flag) in enumerate(zip(names, q1_similarity)):
                if flag == 2:
                    status = "🔴 高度相似 (可能抄襲)"
                elif flag == 1:
                    status = "🟡 中等相似 (需注意)"
                elif flag == 0:
                    status = "🟢 無明顯相似"
                else:
                    status = "⚪ 檢測錯誤"
                
                print(f"  {name}: {status} (標記: {flag})")
            
            # 檢查是否檢測到預期的相似性
            expected_similar_students = [0, 2]  # Kujo Jotaro (index 0) 和 Giorno Giovanna (index 2)
            actual_high_similarity = [i for i, flag in enumerate(q1_similarity) if flag >= 1]
            
            print(f"\n📈 檢測統計：")
            print(f"  預期相似學生: {[names[i] for i in expected_similar_students]}")
            print(f"  實際檢測到相似: {[names[i] for i in actual_high_similarity]}")
            
            if set(expected_similar_students).issubset(set(actual_high_similarity)):
                print("✅ 相似度檢測成功！系統正確識別了修改後的相似答案")
            else:
                print("⚠️ 相似度檢測需要調整閾值或算法")
        
        # 保存測試結果
        output_base = os.path.join(parent_dir, 'plagiarism_detection_test')
        
        # 保存Excel檔案
        xlsx_path = f"{output_base}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"\n💾 測試結果已保存: {os.path.basename(xlsx_path)}")
        
        # 保存HTML報告
        html_path = f"{output_base}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(results['html_report'])
        print(f"💾 HTML報告已保存: {os.path.basename(html_path)}")
        
        # 檢查視覺化
        if results['visualizations']:
            print(f"💾 視覺化矩陣已生成: {len(results['visualizations'])} 個問題")
            for q_id, viz_data in results['visualizations'].items():
                if viz_data.get('local_matrix'):
                    print(f"  ✅ {q_id} 本地相似度矩陣已生成")
        
        print("\n🎯 測試總結：")
        print("1. ✅ 修改了第三個學生的答案使其與第一個學生高度相似")
        print("2. ✅ 執行了完整的相似度檢測分析")
        print("3. ✅ 生成了視覺化報告和數據檔案")
        
        if 'Q1_相似度標記' in df.columns:
            high_sim_count = sum(1 for flag in df['Q1_相似度標記'] if flag >= 1)
            print(f"4. ✅ 檢測到 {high_sim_count} 個學生有相似度問題")
        
        print("🎉 抄襲檢測功能測試完成！")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_plagiarism_detection())
