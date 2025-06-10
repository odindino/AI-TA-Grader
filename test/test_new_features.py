#!/usr/bin/env python3
"""
測試新功能：API key可選 + 視覺化相似度矩陣
"""

import sys
import os

# 添加父目錄到路徑
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'backend'))

def test_without_api_key():
    """測試沒有API key的情況"""
    print("=== 測試無API key功能 ===")
    
    # 模擬前端調用
    from app import Api
    
    api = Api()
    
    # 測試參數（無API key）
    test_params = {
        'apiKey': '',  # 空的API key
        'filePath': '',  # 使用預設測試檔案
        'modelName': 'gemini-1.5-pro-latest'
    }
    
    # 測試開始分析
    result = api.start_analysis(test_params)
    print(f"結果: {result}")
    
    if result['status'] == 'success':
        print("✅ 無API key模式啟動成功")
    else:
        print(f"❌ 無API key模式啟動失敗: {result.get('message', '')}")

def test_enhanced_similarity():
    """測試增強的相似度檢測"""
    print("\n=== 測試增強相似度檢測 ===")
    
    from alternative_similarity_methods import calculate_text_similarity_enhanced
    
    # 測試文本
    test_texts = [
        "The Czochralski method is a crystal growth technique where a seed crystal is dipped into molten silicon.",
        "The Czochralski process is a crystal growing technique in which a seed crystal is immersed in molten silicon.",
        "Molecular beam epitaxy is a completely different thin film deposition technique.",
        "Quantum dots exhibit size-dependent optical properties."
    ]
    
    test_names = ["Alice", "Bob", "Charlie", "David"]
    
    results, details = calculate_text_similarity_enhanced(test_texts, test_names, threshold=0.6)
    
    print(f"相似度結果: {results}")
    print(f"詳細配對:")
    for detail in details:
        print(f"  {detail['student_1']} ↔ {detail['student_2']}: {detail['similarity']:.3f}")

def test_visualization():
    """測試視覺化功能"""
    print("\n=== 測試視覺化功能 ===")
    
    # 測試基本的matplotlib和seaborn導入
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ matplotlib和seaborn導入成功")
        
        # 測試基本繪圖
        import numpy as np
        
        # 創建測試矩陣
        test_matrix = np.random.rand(4, 4)
        np.fill_diagonal(test_matrix, 1.0)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(test_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title('測試相似度矩陣')
        plt.tight_layout()
        
        # 不保存，只是測試是否能正常創建
        plt.close()
        print("✅ 視覺化矩陣創建成功")
        
    except Exception as e:
        print(f"❌ 視覺化測試失敗: {e}")

def main():
    """主測試函數"""
    print("🧪 AI-TA-Grader 新功能測試")
    print("=" * 50)
    
    # 測試1: 無API key功能
    # test_without_api_key()
    
    # 測試2: 增強相似度檢測
    test_enhanced_similarity()
    
    # 測試3: 視覺化功能
    test_visualization()
    
    print("\n✅ 所有測試完成!")

if __name__ == "__main__":
    main()
