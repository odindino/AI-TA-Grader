#!/usr/bin/env python3
"""
直接測試相似度檢測算法
"""

import sys
import os

# 添加路徑
parent_dir = os.path.dirname(__file__)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'backend'))

def test_similarity_algorithms():
    """直接測試相似度檢測算法"""
    print("🔍 直接測試相似度檢測算法")
    print("=" * 50)
    
    from backend.similarity_detector import SimilarityDetector
    
    # 測試文本 - 模擬第一個和第三個學生的答案
    original_answer = """Comparison of the two methods of Czochralski (CZ) process and Floating Zone (FZ) process for single-crystal silicon growth based on advantages and disadvantages as follow below:   
CZ method is more popular used for mass production for Silicon crystal growth was discovered accidently in 1956.  
1. Cheaper
2. Larger wafer size (300 mm in production)
3. Reusable materials
• Floating Zone process is alternative way to make single-crystal ingot.
1. Pure silicon crystal (no crucible)
2. More expensive, smaller wafer size (150 mm)
3. Mainly for power devices

Even though FZ method is more expensive its useful in making smaller wafer size typical 150 nm is primary advantage of FZ is higher Si purity with less crucible contamination.

Channeling effect is phenomena that occur during the incident angle is right, ion can travel long distance without collision with lattice atoms. Also the projected range always follows Gaussian distribution. 
There are many channels along certain angles
Tilt wafer, 7° is most commonly used
Pre-amorphous implantation by Germanium
Rotate wafer and post-implantation diffusion"""

    similar_answer = """Comparison between the Czochralski (CZ) process and Floating Zone (FZ) process for single-crystal silicon growth based on advantages and disadvantages as follow below:
CZ method is more popular used for mass production for Silicon crystal growth was discovered accidently in 1956.
1. Cheaper
2. Larger wafer size (300 mm in production)
3. Reusable materials
• Floating Zone process is alternative way to make single-crystal ingot.
1. Pure silicon crystal (no crucible)
2. More expensive, smaller wafer size (150 mm)  
3. Mainly for power devices

Even though FZ method is more expensive its useful in making smaller wafer size typical 150 nm is primary advantage of FZ is higher Si purity with less crucible contamination.

Channeling effect is phenomena that occur during the incident angle is right, ion can travel long distance without collision with lattice atoms. Also the projected range always follows Gaussian distribution.
There are many channels along certain angles
Tilt wafer, 7° is most commonly used
Pre-amorphous implantation by Germanium
Rotate wafer and post-implantation diffusion"""

    different_answer = """The CZ process melts Si ingot in a crucible, it is much cheaper and is able to fabricate a much larger wafer. The FZ process reorients a polycrystal Si with a heating coil without a crucible. It can achieve higher purity due to the lack of the crucible. But it is more expensive and can only fabricate a small wafer each time. The power devices usually require FZ process despite its higher cost.

During the ion implantation, when a ion beam is applied on the substrate, usually the ion will stop at some point due to the collision. However, with certain incident angles, the ion will go through a much larger distance due to the lack of the collision. This may cause a uncontrollable dopping region size due to the distance of the ion traveling through is much different.

We can use tilted angle method, rotate the wafer when the ion beam is shined, or use the screen oxide method to avoid channeling effect."""

    test_texts = [original_answer, different_answer, similar_answer]
    test_names = ["Kujo Jotaro (原始)", "Joseph Joestar (不同)", "Giorno Giovanna (相似)"]
    
    detector = SimilarityDetector()
    
    print("🔬 測試文本相似度檢測...")
    results = detector.calculate_local_similarity(test_texts, test_names)
    
    print(f"\n📊 相似度檢測結果：")
    print(f"相似度標記: {results['flags']}")
    print(f"檢測方法: {results['info']['method']}")
    print(f"檢測狀態: {results['info']['status']}")
    
    # 分析結果
    print(f"\n📈 詳細分析：")
    for i, (name, flag) in enumerate(zip(test_names, results['flags'])):
        if flag == 2:
            status = "🔴 高度相似 (可能抄襲)"
        elif flag == 1:
            status = "🟡 中等相似 (需注意)"
        elif flag == 0:
            status = "🟢 無明顯相似"
        else:
            status = "⚪ 檢測錯誤"
        
        print(f"  {name}: {status} (標記: {flag})")
    
    # 測試兩兩相似度
    print(f"\n🔍 兩兩相似度測試：")
    from backend.alternative_similarity_methods import calculate_advanced_similarity
    
    pairs = [
        (0, 1, "原始 vs 不同"),
        (0, 2, "原始 vs 相似"), 
        (1, 2, "不同 vs 相似")
    ]
    
    for i, j, desc in pairs:
        similarity = calculate_advanced_similarity(test_texts[i], test_texts[j])
        print(f"  {desc}: {similarity:.3f}")
    
    # 檢查結果是否符合預期
    expected_high_similarity = [0, 2]  # 第1個和第3個學生應該相似
    actual_high_similarity = [i for i, flag in enumerate(results['flags']) if flag >= 1]
    
    print(f"\n🎯 驗證結果：")
    if set(expected_high_similarity).issubset(set(actual_high_similarity)):
        print("✅ 相似度檢測成功！系統正確識別了相似的答案")
    else:
        print("⚠️ 相似度檢測可能需要調整")
        print(f"預期相似: {expected_high_similarity}")
        print(f"實際檢測: {actual_high_similarity}")

if __name__ == "__main__":
    test_similarity_algorithms()
