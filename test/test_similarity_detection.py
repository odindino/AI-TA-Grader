#!/usr/bin/env python3
# test_similarity_detection.py
# 測試相似度檢測功能

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import calculate_similarity_flags, get_detailed_similarity_analysis, load_exam

def test_similarity_detection():
    """測試相似度檢測功能"""
    print("🧪 測試相似度檢測功能")
    print("=" * 50)
    
    # 測試用例：模擬學生答案
    test_cases = [
        {
            "name": "完全不同的答案",
            "answers": [
                "The Czochralski method involves melting silicon and pulling a crystal.",
                "Molecular beam epitaxy uses ultra-high vacuum conditions.",
                "Quantum confinement effects occur in nanoscale materials."
            ],
            "names": ["Alice", "Bob", "Carol"]
        },
        {
            "name": "高度相似的答案（可能抄襲）",
            "answers": [
                "The Czochralski (CZ) method is a crystal growth technique that starts with a seed crystal dipped into molten silicon. The crystal is slowly rotated and pulled upward to form a large single crystal.",
                "The Czochralski (CZ) process is a crystal growth method that begins with a seed crystal inserted into molten silicon. The crystal is gradually rotated and lifted to create a large single crystal.",
                "Molecular beam epitaxy is completely different technique used for thin film growth."
            ],
            "names": ["David", "Eve", "Frank"]
        },
        {
            "name": "中等相似度答案",
            "answers": [
                "Silicon solar cells have efficiency around 20% due to their band gap properties.",
                "The efficiency of silicon photovoltaic cells is approximately 20% because of band gap characteristics.",
                "GaN LEDs have different efficiency characteristics compared to silicon devices."
            ],
            "names": ["Grace", "Henry", "Irene"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 測試案例 {i}: {test_case['name']}")
        print("-" * 40)
        
        # 顯示測試答案
        for j, (name, answer) in enumerate(zip(test_case['names'], test_case['answers']), 1):
            print(f"  {j}. {name}: {answer[:60]}...")
        
        # 執行相似度檢測
        flags = calculate_similarity_flags(
            test_case['answers'], 
            names=test_case['names'],
            hi=0.85,
            mid=0.70,
            min_length=20  # 降低最小長度以適應測試
        )
        
        # 解釋結果
        print(f"\n🔍 相似度檢測結果:")
        for name, flag in zip(test_case['names'], flags):
            if flag == 2:
                status = "🚨 高度相似（可能抄襲）"
            elif flag == 1:
                status = "⚠️  中等相似"
            elif flag == 0:
                status = "✅ 無明顯相似"
            else:
                status = "❌ 檢測錯誤"
            print(f"  {name}: {status}")
        
        # 取得詳細分析
        analysis = get_detailed_similarity_analysis(
            test_case['answers'], 
            names=test_case['names'],
            threshold=0.70
        )
        
        if analysis["status"] == "completed" and analysis["flagged_pairs"]:
            print(f"\n📊 詳細分析:")
            print(f"  總比較次數: {analysis['total_comparisons']}")
            print(f"  標記配對: {analysis['flagged_pairs']}")
            print(f"  高相似度配對:")
            for pair in analysis["high_similarity_pairs"]:
                print(f"    - {pair['student_1']} ↔ {pair['student_2']}: {pair['similarity']:.3f}")

def test_with_real_data():
    """使用真實數據測試"""
    print(f"\n🎯 使用真實數據測試")
    print("=" * 50)
    
    def log_callback(message):
        print(f"LOG: {message}")
    
    # 載入測試檔案
    test_file = "testfile/Final Exam Quiz Student Analysis Report_Public.csv"
    if not os.path.exists(test_file):
        print(f"❌ 測試檔案不存在: {test_file}")
        return
    
    df, q_map = load_exam(test_file, log_callback)
    if df is None or not q_map:
        print("❌ 無法載入測試數據")
        return
    
    # 選擇第一個問題進行測試
    first_qid = list(q_map.keys())[0]
    first_col = q_map[first_qid]
    
    print(f"\n測試問題: Q{first_qid}")
    print(f"問題內容: {first_col[:100]}...")
    
    # 取得答案
    answers = df[first_col].fillna("").tolist()
    names = df["name"].tolist()
    
    print(f"\n學生數量: {len(answers)}")
    
    # 執行相似度檢測
    flags = calculate_similarity_flags(answers, names=names)
    
    # 統計結果
    high_sim = sum(1 for f in flags if f == 2)
    mid_sim = sum(1 for f in flags if f == 1)
    no_sim = sum(1 for f in flags if f == 0)
    errors = sum(1 for f in flags if f == -1)
    
    print(f"\n📈 統計結果:")
    print(f"  高度相似: {high_sim} 人")
    print(f"  中等相似: {mid_sim} 人")
    print(f"  無明顯相似: {no_sim} 人")
    print(f"  檢測錯誤: {errors} 人")
    
    # 詳細分析
    analysis = get_detailed_similarity_analysis(answers, names=names, threshold=0.70)
    if analysis["status"] == "completed":
        print(f"\n📊 詳細分析:")
        print(f"  標記的高相似度配對: {analysis['flagged_pairs']}")
        if analysis["flagged_pairs"] > 0:
            print(f"  配對詳情:")
            for pair in analysis["high_similarity_pairs"][:5]:  # 只顯示前5個
                print(f"    - {pair['student_1']} ↔ {pair['student_2']}: {pair['similarity']:.3f}")

def main():
    print("🔍 相似度檢測功能測試")
    print("=" * 60)
    
    # 測試合成數據
    test_similarity_detection()
    
    # 測試真實數據
    test_with_real_data()
    
    print(f"\n" + "=" * 60)
    print("✅ 測試完成")
    
    print(f"\n💡 改進總結:")
    print("1. ✅ 添加了文本長度過濾（避免過短答案的誤判）")
    print("2. ✅ 提供了詳細的配對資訊（誰和誰相似）")
    print("3. ✅ 改善了文本預處理（清理多餘空白）")
    print("4. ✅ 增加了詳細分析功能（完整相似度矩陣）")
    print("5. ✅ 更好的日誌記錄（記錄高相似度配對）")
    print("6. ✅ 可調整的參數（閾值、最小長度等）")

if __name__ == "__main__":
    main()
