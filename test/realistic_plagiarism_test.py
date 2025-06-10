#!/usr/bin/env python3
# realistic_plagiarism_test.py
# 更實際的抄襲檢測測試案例

from alternative_similarity_methods import comprehensive_plagiarism_check
import re

def test_realistic_cases():
    """測試更實際的抄襲案例"""
    
    print("🧪 實際抄襲檢測測試")
    print("=" * 60)
    
    # 測試案例 1: 明顯抄襲（高度相似）
    case1_texts = [
        """The Czochralski method is a crystal growth technique used to produce single crystals of silicon. In this process, a seed crystal is dipped into molten silicon and slowly pulled upward while rotating. The rotation ensures uniform crystal growth and the pulling speed controls the crystal diameter. This method produces high-quality crystals suitable for semiconductor applications.""",
        
        """The Czochralski process is a crystal growth method used to create single crystals of silicon. In this technique, a seed crystal is immersed in molten silicon and gradually pulled up while rotating. The rotation ensures even crystal formation and the pulling rate controls the crystal size. This approach produces excellent crystals perfect for semiconductor uses.""",
        
        """Molecular beam epitaxy (MBE) is a thin film deposition technique that operates under ultra-high vacuum conditions. It allows precise control of layer thickness and composition by evaporating materials from separate sources. The technique is widely used for creating high-quality heterostructures in optoelectronic devices."""
    ]
    
    case1_names = ["學生A", "學生B", "學生C"]
    
    print("\n📋 測試案例 1: 明顯抄襲案例")
    print("-" * 40)
    for i, (name, text) in enumerate(zip(case1_names, case1_texts), 1):
        print(f"{i}. {name}: {text[:80]}...")
    
    print("\n🔍 檢測結果:")
    comprehensive_plagiarism_check(case1_texts, case1_names)
    
    # 測試案例 2: 完全相同
    case2_texts = [
        """Silicon solar cells achieve efficiency around 20-25% due to their optimal band gap of 1.1 eV.""",
        """Silicon solar cells achieve efficiency around 20-25% due to their optimal band gap of 1.1 eV.""",
        """GaN LEDs have different efficiency characteristics due to their direct bandgap nature."""
    ]
    
    case2_names = ["學生D", "學生E", "學生F"]
    
    print("\n" + "=" * 60)
    print("📋 測試案例 2: 完全相同答案")
    print("-" * 40)
    for i, (name, text) in enumerate(zip(case2_names, case2_texts), 1):
        print(f"{i}. {name}: {text}")
    
    print("\n🔍 檢測結果:")
    comprehensive_plagiarism_check(case2_texts, case2_names)
    
    # 測試案例 3: 中等相似度
    case3_texts = [
        """Silicon photovoltaic cells typically achieve power conversion efficiency between 20% and 25% because silicon has a bandgap energy of approximately 1.1 electron volts, which is well-suited for solar spectrum absorption.""",
        
        """The efficiency of silicon solar cells is usually around 20-25% since silicon possesses a band gap of about 1.1 eV, making it ideal for converting sunlight into electricity.""",
        
        """Gallium arsenide devices exhibit superior performance in high-frequency applications due to their higher electron mobility compared to silicon-based devices."""
    ]
    
    case3_names = ["學生G", "學生H", "學生I"]
    
    print("\n" + "=" * 60)
    print("📋 測試案例 3: 中等相似度（改寫）")
    print("-" * 40)
    for i, (name, text) in enumerate(zip(case3_names, case3_texts), 1):
        print(f"{i}. {name}: {text[:80]}...")
    
    print("\n🔍 檢測結果:")
    comprehensive_plagiarism_check(case3_texts, case3_names)

def test_threshold_sensitivity():
    """測試不同閾值的敏感度"""
    
    print("\n" + "=" * 60)
    print("🎯 閾值敏感度測試")
    print("=" * 60)
    
    # 測試文本
    texts = [
        "The Czochralski method involves pulling a seed crystal from molten silicon while rotating.",
        "The Czochralski process requires pulling a seed from molten silicon with rotation.",
        "Molecular beam epitaxy uses vacuum conditions for thin film growth."
    ]
    names = ["測試A", "測試B", "測試C"]
    
    from alternative_similarity_methods import calculate_text_similarity_local, calculate_tfidf_similarity
    
    # 測試不同閾值
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("\n📊 本地相似度方法測試:")
    for threshold in thresholds:
        results, details = calculate_text_similarity_local(texts, names, threshold)
        flagged_count = sum(1 for r in results if r > 0)
        print(f"  閾值 {threshold}: 標記 {flagged_count} 人")
        for detail in details:
            print(f"    - {detail['student_1']} ↔ {detail['student_2']}: {detail['similarity']:.3f}")
    
    print("\n📊 TF-IDF 相似度方法測試:")
    for threshold in thresholds:
        results, details = calculate_tfidf_similarity(texts, threshold)
        flagged_count = sum(1 for r in results if r > 0)
        print(f"  閾值 {threshold}: 標記 {flagged_count} 人")
        for detail in details:
            print(f"    - {detail['student_1']} ↔ {detail['student_2']}: {detail['similarity']:.3f}")

def provide_usage_recommendations():
    """提供使用建議"""
    
    print("\n" + "=" * 60)
    print("💡 相似度檢測使用建議")
    print("=" * 60)
    
    recommendations = """
🎯 建議的檢測策略：

1. 【主要方法】Google Embeddings (現有實現)
   ✅ 優點：語意理解能力強，能檢測改寫和同義詞替換
   ✅ 適用：大部分抄襲檢測場景
   ⚠️ 限制：需要 API 金鑰，有使用成本

2. 【預檢測】完全相同檢測
   ✅ 用途：快速發現完全複製的答案
   ✅ 效率：計算快速，無需 API
   📋 實現：簡單字符串比較

3. 【備用方法】TF-IDF + 本地相似度
   ✅ 用途：API 不可用時的替代方案
   ✅ 效果：適合檢測詞彙重疊型抄襲
   ⚠️ 限制：對語意理解能力較弱

🔧 推薦的閾值設定：

• Google Embeddings:
  - 高風險: ≥ 0.85 (建議人工審查)
  - 中風險: ≥ 0.70 (重點關注)
  
• TF-IDF 相似度:
  - 高風險: ≥ 0.85 (詞彙高度重疊)
  - 中風險: ≥ 0.70 (詞彙中度重疊)

• 本地文本相似度:
  - 高風險: ≥ 0.80 (結構+詞彙相似)
  - 中風險: ≥ 0.65 (一定程度相似)

📈 檢測流程建議：

1. 預檢測 → 完全相同答案
2. 主檢測 → Google Embeddings
3. 備用檢測 → 本地方法（API 失敗時）
4. 人工審查 → 高風險案例
5. 重點關注 → 中風險案例

🎪 實際應用注意事項：

• 短文本（<50字符）容易產生誤判，建議過濾
• 標準答案模式可能導致正常學生被標記，需要建立白名單
• 不同題目可能需要不同的閾值設定
• 建議保存檢測結果供後續分析和改進
"""
    
    print(recommendations)

if __name__ == "__main__":
    test_realistic_cases()
    test_threshold_sensitivity()
    provide_usage_recommendations()
