#!/usr/bin/env python3
# enhanced_plagiarism_detector.py
# 增強版抄襲檢測器 - 結合多種方法的最佳實踐

import numpy as np
import re
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

class EnhancedPlagiarismDetector:
    """
    增強版抄襲檢測器
    結合 Google Embeddings、TF-IDF、字符相似度等多種方法
    """
    
    def __init__(self, min_text_length=50, high_threshold=0.85, mid_threshold=0.70):
        self.min_text_length = min_text_length
        self.high_threshold = high_threshold
        self.mid_threshold = mid_threshold
        
    def preprocess_text(self, text: str) -> str:
        """標準化文本預處理"""
        if not text or not text.strip():
            return ""
        
        # 1. 基本清理
        cleaned = text.strip()
        
        # 2. 統一空白字符
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 3. 移除多餘標點（可選）
        # cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        
        # 4. 轉換為小寫（用於某些比較）
        # cleaned = cleaned.lower()
        
        return cleaned
    
    def calculate_embedding_similarity(self, texts: List[str], names: List[str] = None) -> Tuple[List[int], List[Dict]]:
        """
        使用 Google Embeddings 計算語意相似度
        """
        try:
            # 預處理
            processed_texts = [self.preprocess_text(t) for t in texts]
            
            # 過濾過短文本
            valid_mask = [len(t) >= self.min_text_length for t in processed_texts]
            if sum(valid_mask) < 2:
                return [0] * len(texts), []
            
            # 計算嵌入向量
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=processed_texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = result['embedding']
            
            # 計算相似度矩陣
            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, 0)
            
            # 分析每個學生的最高相似度
            flags = []
            details = []
            
            for i in range(len(texts)):
                if not valid_mask[i]:
                    flags.append(0)
                    continue
                
                max_sim = np.max(sim_matrix[i])
                max_idx = np.argmax(sim_matrix[i])
                
                # 分類相似度等級
                if max_sim >= self.high_threshold:
                    flag = 2
                elif max_sim >= self.mid_threshold:
                    flag = 1
                else:
                    flag = 0
                
                flags.append(flag)
                
                # 記錄高相似度詳情
                if max_sim >= self.mid_threshold:
                    student_i = names[i] if names else f"學生 {i+1}"
                    student_j = names[max_idx] if names else f"學生 {max_idx+1}"
                    details.append({
                        "method": "embedding",
                        "student_1": student_i,
                        "student_2": student_j,
                        "similarity": float(max_sim),
                        "level": "high" if flag == 2 else "medium",
                        "index_1": i,
                        "index_2": max_idx
                    })
            
            return flags, details
            
        except Exception as e:
            logging.error(f"Embedding 相似度計算錯誤: {e}")
            return [-1] * len(texts), []
    
    def calculate_tfidf_similarity(self, texts: List[str], names: List[str] = None) -> Tuple[List[int], List[Dict]]:
        """
        使用 TF-IDF 計算詞彙相似度
        """
        try:
            if len(texts) < 2:
                return [0] * len(texts), []
            
            # 預處理
            processed_texts = [self.preprocess_text(t) if t else " " for t in texts]
            
            # TF-IDF 向量化
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            
            # 計算相似度矩陣
            sim_matrix = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(sim_matrix, 0)
            
            # 分析結果
            flags = []
            details = []
            
            for i in range(len(texts)):
                if len(processed_texts[i].strip()) < self.min_text_length:
                    flags.append(0)
                    continue
                
                max_sim = np.max(sim_matrix[i])
                max_idx = np.argmax(sim_matrix[i])
                
                if max_sim >= 0.9:  # TF-IDF 閾值較高
                    flag = 2
                elif max_sim >= 0.75:
                    flag = 1
                else:
                    flag = 0
                
                flags.append(flag)
                
                if max_sim >= 0.75:
                    student_i = names[i] if names else f"學生 {i+1}"
                    student_j = names[max_idx] if names else f"學生 {max_idx+1}"
                    details.append({
                        "method": "tfidf",
                        "student_1": student_i,
                        "student_2": student_j,
                        "similarity": float(max_sim),
                        "level": "high" if flag == 2 else "medium",
                        "index_1": i,
                        "index_2": max_idx
                    })
            
            return flags, details
            
        except Exception as e:
            logging.error(f"TF-IDF 相似度計算錯誤: {e}")
            return [-1] * len(texts), []
    
    def calculate_exact_match(self, texts: List[str], names: List[str] = None) -> List[Dict]:
        """
        檢測完全相同或幾乎相同的答案
        """
        exact_matches = []
        processed_texts = [self.preprocess_text(t).lower() for t in texts]
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not processed_texts[i] or not processed_texts[j]:
                    continue
                
                # 檢查完全相同
                if processed_texts[i] == processed_texts[j]:
                    student_i = names[i] if names else f"學生 {i+1}"
                    student_j = names[j] if names else f"學生 {j+1}"
                    exact_matches.append({
                        "method": "exact_match",
                        "student_1": student_i,
                        "student_2": student_j,
                        "similarity": 1.0,
                        "level": "identical",
                        "index_1": i,
                        "index_2": j
                    })
        
        return exact_matches
    
    def comprehensive_analysis(self, texts: List[str], names: List[str] = None) -> Dict:
        """
        綜合分析：結合多種檢測方法
        """
        print("🔍 開始綜合抄襲檢測分析...")
        
        results = {
            "total_students": len(texts),
            "methods_used": [],
            "high_risk_students": set(),
            "medium_risk_students": set(),
            "detailed_findings": [],
            "summary": {}
        }
        
        # 1. 完全相同檢測
        exact_matches = self.calculate_exact_match(texts, names)
        if exact_matches:
            results["methods_used"].append("exact_match")
            results["detailed_findings"].extend(exact_matches)
            for match in exact_matches:
                results["high_risk_students"].add(match["student_1"])
                results["high_risk_students"].add(match["student_2"])
        
        # 2. Embedding 相似度
        try:
            embedding_flags, embedding_details = self.calculate_embedding_similarity(texts, names)
            if any(f >= 0 for f in embedding_flags):
                results["methods_used"].append("embedding")
                results["detailed_findings"].extend(embedding_details)
                
                for i, flag in enumerate(embedding_flags):
                    student = names[i] if names else f"學生 {i+1}"
                    if flag == 2:
                        results["high_risk_students"].add(student)
                    elif flag == 1:
                        results["medium_risk_students"].add(student)
                        
        except Exception as e:
            print(f"⚠️ Embedding 方法失敗: {e}")
        
        # 3. TF-IDF 相似度
        try:
            tfidf_flags, tfidf_details = self.calculate_tfidf_similarity(texts, names)
            if any(f >= 0 for f in tfidf_flags):
                results["methods_used"].append("tfidf")
                results["detailed_findings"].extend(tfidf_details)
                
                for i, flag in enumerate(tfidf_flags):
                    student = names[i] if names else f"學生 {i+1}"
                    if flag == 2:
                        results["high_risk_students"].add(student)
                    elif flag == 1:
                        results["medium_risk_students"].add(student)
                        
        except Exception as e:
            print(f"⚠️ TF-IDF 方法失敗: {e}")
        
        # 轉換 set 為 list
        results["high_risk_students"] = list(results["high_risk_students"])
        results["medium_risk_students"] = list(results["medium_risk_students"])
        
        # 生成摘要
        results["summary"] = {
            "total_methods": len(results["methods_used"]),
            "exact_matches": len(exact_matches),
            "high_risk_count": len(results["high_risk_students"]),
            "medium_risk_count": len(results["medium_risk_students"]),
            "total_flagged_pairs": len(results["detailed_findings"])
        }
        
        return results
    
    def print_analysis_report(self, results: Dict):
        """
        印出詳細的分析報告
        """
        print("\n" + "=" * 60)
        print("📋 綜合抄襲檢測報告")
        print("=" * 60)
        
        summary = results["summary"]
        print(f"📊 檢測概況:")
        print(f"  • 總學生數: {results['total_students']}")
        print(f"  • 使用方法: {', '.join(results['methods_used'])}")
        print(f"  • 完全相同: {summary['exact_matches']} 對")
        print(f"  • 高風險學生: {summary['high_risk_count']} 人")
        print(f"  • 中風險學生: {summary['medium_risk_count']} 人")
        print(f"  • 總可疑配對: {summary['total_flagged_pairs']} 對")
        
        # 詳細發現
        if results["detailed_findings"]:
            print(f"\n🚨 詳細發現:")
            
            # 按相似度排序
            sorted_findings = sorted(
                results["detailed_findings"], 
                key=lambda x: x["similarity"], 
                reverse=True
            )
            
            for finding in sorted_findings:
                method = finding["method"]
                level = finding["level"]
                sim = finding["similarity"]
                
                emoji = {"identical": "🔴", "high": "🟠", "medium": "🟡"}[level]
                print(f"  {emoji} [{method.upper()}] {finding['student_1']} ↔ {finding['student_2']}")
                print(f"     相似度: {sim:.3f} ({level})")
        
        # 風險學生列表
        if results["high_risk_students"]:
            print(f"\n🎯 高風險學生列表:")
            for student in sorted(results["high_risk_students"]):
                print(f"  🚨 {student}")
        
        if results["medium_risk_students"]:
            print(f"\n⚠️ 中風險學生列表:")
            for student in sorted(results["medium_risk_students"]):
                print(f"  ⚠️ {student}")
        
        if not results["high_risk_students"] and not results["medium_risk_students"]:
            print(f"\n✅ 未檢測到明顯的抄襲行為")


def test_enhanced_detector():
    """測試增強版檢測器"""
    
    # 測試案例
    test_texts = [
        "The Czochralski method is a crystal growth technique where a seed crystal is dipped into molten silicon and slowly pulled upward while rotating to form a single crystal ingot.",
        "The Czochralski process is a crystal growing technique in which a seed crystal is immersed in molten silicon and gradually pulled up while rotating to create a single crystal ingot.",
        "Molecular beam epitaxy (MBE) is a thin film deposition technique that allows precise control of layer thickness and composition by evaporating materials in ultra-high vacuum.",
        "Silicon solar cells achieve efficiency around 20-25% due to their optimal band gap of 1.1 eV, making them suitable for converting solar energy into electrical energy.",
        "Gallium arsenide (GaAs) devices have different optical and electronic properties compared to silicon, including a direct bandgap that makes them efficient for optoelectronic applications."
    ]
    
    test_names = ["Alice", "Bob", "Charlie", "David", "Eve"]
    
    print("🧪 測試增強版抄襲檢測器")
    print("=" * 50)
    
    # 創建檢測器實例
    detector = EnhancedPlagiarismDetector(
        min_text_length=30,
        high_threshold=0.85,
        mid_threshold=0.70
    )
    
    # 執行綜合分析
    results = detector.comprehensive_analysis(test_texts, test_names)
    
    # 印出報告
    detector.print_analysis_report(results)
    
    return results


if __name__ == "__main__":
    # 需要先設定 Google API
    # genai.configure(api_key="your_api_key_here")
    
    test_enhanced_detector()
