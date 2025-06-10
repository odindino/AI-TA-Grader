#!/usr/bin/env python3
"""
完整抄襲檢測測試 - 包含視覺化相似度矩陣
"""

import sys
import os
import asyncio

# 添加路徑
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'backend'))

async def run_plagiarism_analysis():
    """執行完整的抄襲檢測分析"""
    print("🕵️ AI-TA-Grader 抄襲檢測完整測試")
    print("=" * 60)
    
    from backend.analyzer import AnalysisEngine
    
    # 測試檔案路徑
    test_csv = os.path.join(parent_dir, 'testfile', 'Final Exam Quiz Student Analysis Report_Public_plag.csv')
    
    if not os.path.exists(test_csv):
        print(f"❌ 測試檔案不存在: {test_csv}")
        return
    
    print(f"📁 使用測試檔案: {os.path.basename(test_csv)}")
    
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
        
        # 檢查結果
        df = results['dataframe']
        visualizations = results['visualizations']
        html_report = results['html_report']
        
        print(f"\n📈 分析結果摘要：")
        print(f"  總學生數: {len(df)}")
        print(f"  數據欄位: {len(df.columns)}")
        
        # 檢查相似度檢測結果
        similarity_columns = [col for col in df.columns if '_相似度標記' in col]
        if similarity_columns:
            print(f"  相似度檢測欄位: {similarity_columns}")
            
            for col in similarity_columns:
                flags = df[col].tolist()
                high_similarity = sum(1 for flag in flags if flag == 2)
                medium_similarity = sum(1 for flag in flags if flag == 1)
                no_similarity = sum(1 for flag in flags if flag == 0)
                
                print(f"\n🔍 {col} 檢測結果：")
                print(f"    🔴 高度相似: {high_similarity} 位學生")
                print(f"    🟡 中等相似: {medium_similarity} 位學生") 
                print(f"    🟢 無明顯相似: {no_similarity} 位學生")
                
                # 顯示具體的相似度標記
                names = df['name'].tolist() if 'name' in df.columns else [f"學生{i+1}" for i in range(len(df))]
                print(f"  📝 詳細結果：")
                for i, (name, flag) in enumerate(zip(names, flags)):
                    if flag == 2:
                        status = "🔴 高度相似"
                    elif flag == 1:
                        status = "🟡 中等相似"
                    elif flag == 0:
                        status = "🟢 無相似"
                    else:
                        status = f"⚪ 錯誤({flag})"
                    print(f"    {name}: {status}")
        
        # 檢查視覺化結果
        if visualizations:
            print(f"\n🎨 視覺化矩陣生成：")
            for q_id, viz_data in visualizations.items():
                print(f"  📊 {q_id}:")
                if viz_data.get('genai_matrix'):
                    print(f"    ✅ GenAI相似度矩陣已生成")
                if viz_data.get('local_matrix'):
                    print(f"    ✅ 本地相似度矩陣已生成")
        else:
            print("⚠️ 未生成視覺化矩陣")
        
        # 保存結果
        output_base = os.path.join(parent_dir, 'plagiarism_analysis_complete')
        
        print(f"\n💾 保存分析結果...")
        
        # 保存Excel檔案
        xlsx_path = f"{output_base}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"  ✅ Excel報告: {os.path.basename(xlsx_path)}")
        
        # 保存CSV檔案
        csv_path = f"{output_base}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✅ CSV檔案: {os.path.basename(csv_path)}")
        
        # 保存HTML報告
        html_path = f"{output_base}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"  ✅ HTML報告: {os.path.basename(html_path)}")
        print(f"    📊 報告長度: {len(html_report):,} 字符")
        
        # 檢查HTML中是否包含base64圖像
        import re
        base64_images = re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', html_report)
        if base64_images:
            print(f"    🖼️ 包含 {len(base64_images)} 個視覺化圖像")
        else:
            print(f"    ⚠️ HTML報告中未找到視覺化圖像")
        
        print(f"\n🎯 測試結論：")
        
        # 檢查是否成功檢測到我們預期的抄襲案例
        q1_col = 'Q1_相似度標記'
        if q1_col in df.columns:
            q1_flags = df[q1_col].tolist()
            if len(q1_flags) >= 3:
                # 檢查第1個和第3個學生是否被標記為相似
                student1_flag = q1_flags[0]  # Kujo Jotaro
                student3_flag = q1_flags[2] if len(q1_flags) > 2 else -1  # Giorno Giovanna
                
                if student1_flag >= 1 and student3_flag >= 1:
                    print("✅ 成功檢測到預期的抄襲案例！")
                    print(f"   第1個學生標記: {student1_flag}")
                    print(f"   第3個學生標記: {student3_flag}")
                else:
                    print("⚠️ 未檢測到預期的抄襲案例")
                    print(f"   第1個學生標記: {student1_flag}")
                    print(f"   第3個學生標記: {student3_flag}")
        
        if visualizations:
            print("✅ 視覺化相似度矩陣已生成")
        else:
            print("❌ 視覺化矩陣生成失敗")
        
        if base64_images:
            print("✅ HTML報告包含視覺化圖像")
        else:
            print("❌ HTML報告缺少視覺化圖像")
        
        print(f"\n🎉 抄襲檢測分析完成！")
        print(f"📁 結果檔案位於: {os.path.dirname(output_base)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(run_plagiarism_analysis())
    if success:
        print("\n🏆 測試成功完成！")
    else:
        print("\n💥 測試失敗！")
