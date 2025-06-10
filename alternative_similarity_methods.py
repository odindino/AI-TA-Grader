#!/usr/bin/env python3
# alternative_similarity_methods.py
# 提供不依賴 API 的相似度檢測方法

import re
import difflib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_text_similarity_local(texts: list[str], names: list[str] = None, threshold=0.6):
    """
    使用本地方法計算文本相似度（不依賴 API）
    結合多種相似度指標來檢測潛在抄襲
    """
    if len(texts) < 2:
        return [0] * len(texts)
    
    # 預處理文本
    processed_texts = []
    for text in texts:
        if text and text.strip():
            # 基本文本清理
            cleaned = re.sub(r'\s+', ' ', text.strip().lower())
            # 移除標點符號（保留字母數字）
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            processed_texts.append(cleaned)
        else:
            processed_texts.append("")
    
    results = []
    detailed_results = []
    
    for i, text_i in enumerate(processed_texts):
        if not text_i.strip():
            results.append(0)
            continue
            
        max_similarity = 0
        best_match_idx = -1
        
        for j, text_j in enumerate(processed_texts):
            if i == j or not text_j.strip():
                continue
                
            # 方法 1: 字符級別相似度 (difflib)
            char_similarity = difflib.SequenceMatcher(None, text_i, text_j).ratio()
            
            # 方法 2: 詞彙重疊相似度
            words_i = set(text_i.split())
            words_j = set(text_j.split())
            if words_i or words_j:
                jaccard_similarity = len(words_i & words_j) / len(words_i | words_j)
            else:
                jaccard_similarity = 0
            
            # 方法 3: N-gram 相似度
            ngram_similarity = calculate_ngram_similarity(text_i, text_j, n=3)
            
            # 綜合相似度（加權平均）
            combined_similarity = (
                char_similarity * 0.3 + 
                jaccard_similarity * 0.4 + 
                ngram_similarity * 0.3
            )
            
            if combined_similarity > max_similarity:
                max_similarity = combined_similarity
                best_match_idx = j
        
        # 記錄詳細結果
        if max_similarity >= threshold and best_match_idx >= 0:
            student_i = names[i] if names else f"學生 {i+1}"
            student_j = names[best_match_idx] if names else f"學生 {best_match_idx+1}"
            detailed_results.append({
                "student_1": student_i,
                "student_2": student_j,
                "similarity": max_similarity,
                "index_1": i,
                "index_2": best_match_idx
            })
        
        # 分類相似度等級（調整閾值使其更敏感）
        if max_similarity >= 0.75:
            results.append(2)  # 高度相似
        elif max_similarity >= threshold:
            results.append(1)  # 中等相似
        else:
            results.append(0)  # 無明顯相似
    
    return results, detailed_results

def calculate_ngram_similarity(text1: str, text2: str, n: int = 3):
    """計算 N-gram 相似度"""
    def get_ngrams(text, n):
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    ngrams1 = set(get_ngrams(text1, n))
    ngrams2 = set(get_ngrams(text2, n))
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0

def calculate_tfidf_similarity(texts: list[str], threshold=0.6):
    """使用 TF-IDF 向量化計算相似度"""
    if len(texts) < 2:
        return [0] * len(texts), []
    
    # 過濾空文本
    valid_texts = [text if text and text.strip() else " " for text in texts]
    
    try:
        # TF-IDF 向量化
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        
        # 計算餘弦相似度
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        
        # 分析結果
        results = []
        detailed_results = []
        
        for i in range(len(texts)):
            max_sim = np.max(similarity_matrix[i])
            max_idx = np.argmax(similarity_matrix[i])
            
            if max_sim >= threshold:
                detailed_results.append({
                    "student_1": f"學生 {i+1}",
                    "student_2": f"學生 {max_idx+1}",
                    "similarity": float(max_sim),
                    "index_1": i,
                    "index_2": max_idx
                })
                 # 分類（調整閾值使其更敏感）
        if max_sim >= 0.80:  # TF-IDF 閾值降低
            results.append(2)
        elif max_sim >= threshold:
            results.append(1)
        else:
            results.append(0)
        
        return results, detailed_results
        
    except Exception as e:
        print(f"TF-IDF 計算錯誤: {e}")
        return [0] * len(texts), []

def comprehensive_plagiarism_check(texts: list[str], names: list[str] = None):
    """
    綜合抄襲檢測：結合多種方法
    """
    print("🔍 綜合抄襲檢測分析")
    print("=" * 50)
    
    if len(texts) < 2:
        print("❌ 文本數量不足，無法進行比較")
        return
    
    # 方法 1: 本地文本相似度
    print("\n📝 方法 1: 本地文本相似度分析")
    local_results, local_details = calculate_text_similarity_local(texts, names, threshold=0.7)
    
    print(f"檢測結果:")
    for i, (name, result) in enumerate(zip(names if names else [f"學生 {i+1}" for i in range(len(texts))], local_results)):
        status = ["✅ 無相似", "⚠️ 中等相似", "🚨 高度相似"][result] if result >= 0 else "❌ 錯誤"
        print(f"  {name}: {status}")
    
    if local_details:
        print(f"高相似度配對:")
        for detail in local_details:
            print(f"  - {detail['student_1']} ↔ {detail['student_2']}: {detail['similarity']:.3f}")
    
    # 方法 2: TF-IDF 相似度
    print(f"\n📊 方法 2: TF-IDF 向量相似度分析")
    tfidf_results, tfidf_details = calculate_tfidf_similarity(texts, threshold=0.7)
    
    print(f"檢測結果:")
    for i, (name, result) in enumerate(zip(names if names else [f"學生 {i+1}" for i in range(len(texts))], tfidf_results)):
        status = ["✅ 無相似", "⚠️ 中等相似", "🚨 高度相似"][result] if result >= 0 else "❌ 錯誤"
        print(f"  {name}: {status}")
    
    if tfidf_details:
        print(f"高相似度配對:")
        for detail in tfidf_details:
            print(f"  - {detail['student_1']} ↔ {detail['student_2']}: {detail['similarity']:.3f}")
    
    # 綜合分析
    print(f"\n🎯 綜合分析:")
    high_risk_students = set()
    
    for i, (local, tfidf) in enumerate(zip(local_results, tfidf_results)):
        if local >= 2 or tfidf >= 2:  # 任一方法檢測到高相似度
            student_name = names[i] if names else f"學生 {i+1}"
            high_risk_students.add(student_name)
    
    if high_risk_students:
        print(f"🚨 高風險學生（可能涉及抄襲）:")
        for student in high_risk_students:
            print(f"  - {student}")
    else:
        print(f"✅ 未檢測到明顯的抄襲行為")
    
    return {
        "local_results": local_results,
        "tfidf_results": tfidf_results,
        "high_risk_students": list(high_risk_students),
        "local_details": local_details,
        "tfidf_details": tfidf_details
    }

# 測試函數
def test_alternative_methods():
    """測試替代相似度檢測方法"""
    
    # 測試案例
    test_cases = [
        {
            "name": "明顯抄襲案例",
            "texts": [
                "The Czochralski method is a crystal growth technique where a seed crystal is dipped into molten silicon and slowly pulled upward while rotating.",
                "The Czochralski process is a crystal growing technique in which a seed crystal is immersed in molten silicon and gradually pulled up while rotating.",
                "Molecular beam epitaxy is a completely different thin film deposition technique."
            ],
            "names": ["Alice", "Bob", "Charlie"]
        },
        {
            "name": "正常差異案例",
            "texts": [
                "Silicon solar cells achieve efficiency around 20% due to optimal band gap.",
                "Gallium arsenide devices have different optical properties than silicon.",
                "Quantum dots exhibit size-dependent emission wavelengths."
            ],
            "names": ["David", "Eve", "Frank"]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"📋 測試案例 {i}: {case['name']}")
        print(f"{'='*60}")
        
        for j, (name, text) in enumerate(zip(case['names'], case['texts']), 1):
            print(f"{j}. {name}: {text}")
        
        comprehensive_plagiarism_check(case['texts'], case['names'])

if __name__ == "__main__":
    test_alternative_methods()
