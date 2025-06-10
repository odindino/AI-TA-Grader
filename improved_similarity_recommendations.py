#!/usr/bin/env python3
# improved_similarity_recommendations.py
# 相似度檢測改進建議和實用工具

"""
針對 AI-TA-Grader 的相似度檢測改進建議

=== 當前實現評估 ===
✅ 優秀的基礎：
- Google text-embedding-004 模型（語意理解能力強）
- 合理的閾值設定（0.85 高風險，0.70 中風險）
- 文本長度篩選（50 字符最小值）
- 詳細的配對信息和日誌記錄
- 適當的錯誤處理

=== 推薦的改進策略 ===

1. 【保持現有方法為主】- 繼續使用 Google Embeddings
2. 【添加預檢測層】- 快速過濾明顯案例
3. 【增強報告功能】- 提供更詳細的分析
4. 【備用方法】- 當 API 不可用時的替代方案

=== 具體改進建議 ===
"""

import re
import logging
from typing import List, Dict, Tuple
import numpy as np
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ImprovedSimilarityDetector:
    """
    改進版相似度檢測器 - 基於現有代碼的增強版本
    """
    
    def __init__(self):
        self.min_length = 50
        self.high_threshold = 0.85
        self.mid_threshold = 0.70
    
    def quick_precheck(self, texts: List[str]) -> Dict:
        """
        快速預檢：識別明顯的相同或近似相同答案
        在進行 API 調用前先過濾明顯案例
        """
        precheck_results = {
            "identical_pairs": [],
            "very_similar_pairs": [],
            "short_texts": [],
            "empty_texts": []
        }
        
        processed = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                precheck_results["empty_texts"].append(i)
                processed.append("")
            elif len(text.strip()) < self.min_length:
                precheck_results["short_texts"].append(i)
                processed.append(text.strip().lower())
            else:
                # 標準化處理
                cleaned = re.sub(r'\s+', ' ', text.strip().lower())
                processed.append(cleaned)
        
        # 檢查完全相同
        for i in range(len(processed)):
            for j in range(i + 1, len(processed)):
                if processed[i] and processed[j]:
                    if processed[i] == processed[j]:
                        precheck_results["identical_pairs"].append((i, j))
                    elif self._simple_similarity(processed[i], processed[j]) > 0.95:
                        precheck_results["very_similar_pairs"].append((i, j))
        
        return precheck_results
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """簡單的文本相似度計算（詞彙重疊）"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def enhanced_embedding_similarity(self, texts: List[str], names: List[str] = None) -> Dict:
        """
        增強版 Embedding 相似度檢測
        基於現有代碼，增加更詳細的分析和報告
        """
        # 預檢測
        precheck = self.quick_precheck(texts)
        
        # 準備結果結構
        results = {
            "method": "enhanced_embedding",
            "precheck": precheck,
            "similarity_flags": [],
            "detailed_pairs": [],
            "similarity_matrix": None,
            "statistics": {}
        }
        
        try:
            # 文本預處理（與原代碼一致）
            processed_texts = []
            for text in texts:
                if text and text.strip():
                    cleaned = re.sub(r'\s+', ' ', text.strip())
                    processed_texts.append(cleaned)
                else:
                    processed_texts.append(" ")
            
            # 檢查有效文本數量
            valid_texts = [t for t in processed_texts if len(t.strip()) >= self.min_length]
            if len(valid_texts) < 2:
                results["similarity_flags"] = [0] * len(texts)
                results["statistics"]["status"] = "insufficient_texts"
                return results
            
            # 計算 Embedding（使用與原代碼相同的模型）
            embedding_result = genai.embed_content(
                model="models/text-embedding-004",
                content=processed_texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = embedding_result['embedding']
            
            # 計算相似度矩陣
            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, 0)
            results["similarity_matrix"] = sim_matrix.tolist()
            
            # 分析每個學生
            similarity_flags = []
            detailed_pairs = []
            
            for i in range(len(texts)):
                if len(processed_texts[i].strip()) < self.min_length:
                    similarity_flags.append(0)
                    continue
                
                max_sim = np.max(sim_matrix[i])
                max_idx = np.argmax(sim_matrix[i])
                
                # 分類（使用原代碼的閾值）
                if max_sim >= self.high_threshold:
                    flag = 2
                elif max_sim >= self.mid_threshold:
                    flag = 1
                else:
                    flag = 0
                
                similarity_flags.append(flag)
                
                # 記錄詳細信息
                if max_sim >= self.mid_threshold:
                    student_i = names[i] if names else f"學生 {i+1}"
                    student_j = names[max_idx] if names else f"學生 {max_idx+1}"
                    
                    detailed_pairs.append({
                        "student_1": student_i,
                        "student_2": student_j,
                        "similarity": float(max_sim),
                        "risk_level": "high" if flag == 2 else "medium",
                        "index_1": i,
                        "index_2": max_idx,
                        "text_length_1": len(processed_texts[i]),
                        "text_length_2": len(processed_texts[max_idx])
                    })
            
            results["similarity_flags"] = similarity_flags
            results["detailed_pairs"] = detailed_pairs
            
            # 統計信息
            results["statistics"] = {
                "status": "completed",
                "total_students": len(texts),
                "valid_texts": len(valid_texts),
                "high_risk_count": similarity_flags.count(2),
                "medium_risk_count": similarity_flags.count(1),
                "safe_count": similarity_flags.count(0),
                "error_count": similarity_flags.count(-1),
                "average_similarity": float(np.mean(sim_matrix[sim_matrix > 0])) if np.any(sim_matrix > 0) else 0.0,
                "max_similarity": float(np.max(sim_matrix)),
                "identical_pairs": len(precheck["identical_pairs"]),
                "very_similar_pairs": len(precheck["very_similar_pairs"])
            }
            
        except Exception as e:
            logging.error(f"Enhanced embedding similarity error: {e}")
            results["similarity_flags"] = [-1] * len(texts)
            results["statistics"]["status"] = "error"
            results["statistics"]["error"] = str(e)
        
        return results
    
    def generate_detailed_report(self, results: Dict, names: List[str] = None) -> str:
        """
        生成詳細的檢測報告
        """
        report = ["=" * 60]
        report.append("📋 相似度檢測詳細報告")
        report.append("=" * 60)
        
        stats = results["statistics"]
        
        # 總體統計
        report.append(f"\n📊 檢測統計:")
        report.append(f"  • 總學生數: {stats.get('total_students', 0)}")
        report.append(f"  • 有效文本: {stats.get('valid_texts', 0)}")
        report.append(f"  • 檢測狀態: {stats.get('status', 'unknown')}")
        
        if stats.get('status') == 'completed':
            report.append(f"  • 高風險: {stats.get('high_risk_count', 0)} 人")
            report.append(f"  • 中風險: {stats.get('medium_risk_count', 0)} 人")
            report.append(f"  • 安全: {stats.get('safe_count', 0)} 人")
            report.append(f"  • 平均相似度: {stats.get('average_similarity', 0):.3f}")
            report.append(f"  • 最高相似度: {stats.get('max_similarity', 0):.3f}")
        
        # 預檢測結果
        precheck = results.get("precheck", {})
        if precheck.get("identical_pairs"):
            report.append(f"\n🔴 完全相同的答案:")
            for i, j in precheck["identical_pairs"]:
                name_i = names[i] if names else f"學生 {i+1}"
                name_j = names[j] if names else f"學生 {j+1}"
                report.append(f"  • {name_i} ↔ {name_j}")
        
        # 詳細配對
        detailed_pairs = results.get("detailed_pairs", [])
        if detailed_pairs:
            report.append(f"\n🚨 相似度配對詳情:")
            sorted_pairs = sorted(detailed_pairs, key=lambda x: x["similarity"], reverse=True)
            
            for pair in sorted_pairs:
                emoji = "🔴" if pair["risk_level"] == "high" else "🟡"
                report.append(f"  {emoji} {pair['student_1']} ↔ {pair['student_2']}")
                report.append(f"     相似度: {pair['similarity']:.3f} ({pair['risk_level']})")
                report.append(f"     文本長度: {pair['text_length_1']} vs {pair['text_length_2']} 字符")
        
        # 建議
        report.append(f"\n💡 檢測建議:")
        if stats.get('high_risk_count', 0) > 0:
            report.append("  • 🚨 發現高風險相似答案，建議人工詳細審查")
        if stats.get('medium_risk_count', 0) > 0:
            report.append("  • ⚠️ 發現中風險相似答案，建議重點關注")
        if stats.get('identical_pairs', 0) > 0:
            report.append("  • 🔴 發現完全相同答案，強烈建議調查")
        if stats.get('status') == 'completed' and stats.get('high_risk_count', 0) == 0:
            report.append("  • ✅ 未發現明顯抄襲行為")
        
        return "\n".join(report)

# === 使用建議 ===

def implementation_recommendations():
    """
    實施建議：如何在現有系統中應用這些改進
    """
    print("""
🎯 實施建議：

1. 【短期改進】- 增強現有函數
   • 在 calculate_similarity_flags() 中添加預檢測
   • 增加更詳細的日誌輸出
   • 提供配對詳情的 JSON 輸出

2. 【中期改進】- 添加備用方法
   • 當 API 失敗時，自動切換到 TF-IDF 方法
   • 添加完全相同檢測作為第一道防線
   • 提供離線模式

3. 【長期改進】- 智能化檢測
   • 根據題目類型調整閾值
   • 學習常見的正確答案模式
   • 提供相似度分布分析

4. 【報告改進】- 更好的用戶體驗
   • 在前端顯示相似度配對
   • 提供可下載的檢測報告
   • 添加相似文本的並排比較視圖

=== 代碼整合示例 ===

# 在 analyzer.py 中的簡單改進：

def enhanced_calculate_similarity_flags(texts, names=None, hi=0.85, mid=0.70, min_length=50):
    \"\"\"改進版相似度檢測 - 基於現有代碼\"\"\"
    
    # 1. 預檢測階段
    identical_pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i] and texts[j] and texts[i].strip().lower() == texts[j].strip().lower():
                identical_pairs.append((i, j))
                logging.warning(f"發現完全相同答案: {names[i] if names else f'學生{i+1}'} ↔ {names[j] if names else f'學生{j+1}'}")
    
    # 2. 原有的 embedding 檢測
    # ... 現有代碼 ...
    
    # 3. 增強報告
    if high_similarity_pairs:
        logging.info(f"相似度檢測摘要:")
        logging.info(f"  • 完全相同: {len(identical_pairs)} 對")
        logging.info(f"  • 高相似度: {len([p for p in high_similarity_pairs if '相似度: 0.8' in p])} 對")
        logging.info(f"  • 平均相似度: {np.mean([...]):.3f}")
    
    return similarity_flags

""")

if __name__ == "__main__":
    implementation_recommendations()
