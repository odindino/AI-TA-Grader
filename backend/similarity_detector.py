"""
相似度檢測模組 - 統一的相似度檢測接口
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

from .config import SIMILARITY_THRESHOLDS, DO_SIMILARITY_CHECK


class SimilarityDetector:
    """相似度檢測器類別"""
    
    def __init__(self):
        """初始化相似度檢測器"""
        self.logger = logging.getLogger(__name__)
        self.thresholds = SIMILARITY_THRESHOLDS
    
    async def calculate_genai_similarity(self, texts: List[str], names: List[str] = None, 
                                       gemini_client=None) -> Dict[str, Any]:
        """使用GenAI計算相似度
        
        Args:
            texts: 文本列表
            names: 姓名列表
            gemini_client: Gemini客戶端實例
            
        Returns:
            Dict: 相似度結果
        """
        try:
            if not gemini_client or not gemini_client.model:
                raise ValueError("Gemini客戶端未提供或未初始化")
            
            # 使用Google嵌入API
            processed_texts = [t.strip() if t and t.strip() else " " for t in texts]
            
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=processed_texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            embs = np.array(result['embedding'])
            similarity_matrix = cosine_similarity(embs)
            
            # 計算相似度標記
            flags = self._calculate_flags_from_matrix(similarity_matrix, texts)
            
            # 記錄相似度檢測結果
            self.logger.info(f"🔍 GenAI 相似度檢測結果:")
            for i, flag in enumerate(flags):
                if flag > 0:
                    max_sim = max(similarity_matrix[i][j] for j in range(len(texts)) if i != j) if len(texts) > 1 else 0
                    self.logger.info(f"  學生 {i+1}: 標記={flag}, 最高相似度={max_sim:.3f}")
            
            return {
                'flags': flags,
                'matrix': similarity_matrix,
                'info': {
                    'method': 'genai',
                    'model': 'text-embedding-004',
                    'status': 'success'
                }
            }
            
        except Exception as e:
            self.logger.error(f"GenAI相似度計算失敗: {e}")
            # 降級到本地方法
            return self.calculate_local_similarity(texts, names)
    
    def calculate_local_similarity(self, texts: List[str], names: List[str] = None) -> Dict[str, Any]:
        """使用本地算法計算相似度
        
        Args:
            texts: 文本列表
            names: 姓名列表
            
        Returns:
            Dict: 相似度結果
        """
        try:
            from .alternative_similarity_methods import calculate_advanced_similarity
            
            n = len(texts)
            similarity_matrix = np.zeros((n, n))
            
            # 計算相似度矩陣
            for i in range(n):
                for j in range(n):
                    if i != j and texts[i].strip() and texts[j].strip():
                        similarity = calculate_advanced_similarity(texts[i], texts[j])
                        similarity_matrix[i][j] = similarity
                    elif i == j:
                        similarity_matrix[i][j] = 1.0
            
            # 計算相似度分數 (0-100)
            scores = self._calculate_scores_from_matrix(similarity_matrix, texts)
            
            # 記錄本地相似度檢測結果
            self.logger.info(f"🔍 本地相似度檢測結果:")
            for i, score in enumerate(scores):
                if score > 0:
                    max_sim = max(similarity_matrix[i][j] for j in range(len(texts)) if i != j) if len(texts) > 1 else 0
                    self.logger.info(f"  學生 {i+1}: 相似度分數={score}, 最高相似度={max_sim:.3f}")
            
            return {
                'scores': scores,
                'matrix': similarity_matrix,
                'info': {
                    'method': 'local',
                    'algorithms': 'advanced_multi_algorithm',
                    'status': 'success'
                }
            }
            
        except Exception as e:
            self.logger.error(f"本地相似度計算失敗: {e}")
            return {
                'scores': [0] * len(texts),
                'matrix': None,
                'info': {
                    'method': 'error',
                    'status': 'failed',
                    'error': str(e)
                }
            }
    
    def _calculate_flags_from_matrix(self, similarity_matrix: np.ndarray, texts: List[str]) -> List[int]:
        """從相似度矩陣計算標記"""
        n = len(texts)
        flags = [0] * n
        
        hi_threshold = self.thresholds.get('high', 0.85)
        mid_threshold = self.thresholds.get('medium', 0.7)
        min_length = self.thresholds.get('min_length', 50)
        
        for i in range(n):
            # 檢查文本長度
            if len(texts[i].strip()) < min_length:
                continue
            
            max_similarity = 0.0
            for j in range(n):
                if i != j:
                    max_similarity = max(max_similarity, similarity_matrix[i][j])
            
            # 設定標記
            if max_similarity >= hi_threshold:
                flags[i] = 2  # 高度相似
            elif max_similarity >= mid_threshold:
                flags[i] = 1  # 中等相似
        
        return flags
    
    def _calculate_scores_from_matrix(self, similarity_matrix: np.ndarray, texts: List[str]) -> List[int]:
        """從相似度矩陣計算0-100分的相似度分數"""
        n = len(texts)
        scores = [0] * n
        
        min_length = self.thresholds.get('min_length', 50)
        
        for i in range(n):
            # 檢查文本長度
            if len(texts[i].strip()) < min_length:
                scores[i] = 0
                continue
            
            max_similarity = 0.0
            for j in range(n):
                if i != j:
                    max_similarity = max(max_similarity, similarity_matrix[i][j])
            
            # 將相似度轉換為0-100分
            score = int(max_similarity * 100)
            scores[i] = score
        
        return scores


# 保持向後相容的函數接口
def calculate_similarity_flags(texts: List[str], names: List[str] = None, hi: float = 0.85, mid: float = 0.70, min_length: int = 50, use_api: bool = True) -> Tuple[List[int], dict]:
    """
    計算答案間的相似度，檢測潛在抄襲 - 向後相容函數
    支援 GenAI (需要API) 和非GenAI方法 (本地計算)
    
    Args:
        texts: 學生答案列表
        names: 學生姓名列表（可選）
        hi: 高相似度閾值（預設 0.85）
        mid: 中等相似度閾值（預設 0.70）
        min_length: 最小文本長度閾值（預設 50 字符）
        use_api: 是否使用GenAI API (預設 True)
    
    Returns:
        tuple: (similarity_flags, detailed_results)
               similarity_flags: list[int] - 相似度標記 (0=無相似, 1=中等相似, 2=高度相似, -1=錯誤)
               detailed_results: dict - 詳細分析結果
    """
    if not DO_SIMILARITY_CHECK or len(texts) < 2:
        return [0] * len(texts), {"status": "skipped", "method": "none"}
    
    # 使用配置檔案中的閾值
    thresholds = SIMILARITY_THRESHOLDS
    hi = thresholds.get('high', hi)
    mid = thresholds.get('medium', mid)
    min_length = thresholds.get('min_length', min_length)
    
    if use_api:
        # 使用 GenAI API 方法
        try:
            # 預處理文本
            processed_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip() and len(text.strip()) >= min_length:
                    # 簡單的文本清理
                    cleaned_text = text.strip()
                    # 移除多餘的空白字符
                    cleaned_text = ' '.join(cleaned_text.split())
                    processed_texts.append(cleaned_text)
                    valid_indices.append(i)
                else:
                    processed_texts.append(" ")
                    valid_indices.append(i)
            
            # 如果有效文本少於 2 個，跳過相似度檢查
            valid_texts = [processed_texts[i] for i in range(len(processed_texts)) if len(processed_texts[i].strip()) >= min_length]
            if len(valid_texts) < 2:
                return [0] * len(texts), {"status": "insufficient_data", "method": "genai"}
            
            # 計算語意嵌入
            result = genai.embed_content(
                model="models/text-embedding-004", 
                content=processed_texts, 
                task_type="RETRIEVAL_DOCUMENT"
            )
            embs = result['embedding']
            
            # 計算相似度矩陣
            sims = cosine_similarity(embs)
            np.fill_diagonal(sims, 0)  # 設置對角線為 0，避免自己和自己比較
            
            # 分析相似度結果
            similarity_flags = []
            high_similarity_pairs = []
            
            for i in range(len(texts)):
                if len(processed_texts[i].strip()) < min_length:
                    # 文本過短，不參與相似度檢查
                    similarity_flags.append(0)
                    continue
                    
                max_sim = np.max(sims[i])
                max_sim_idx = np.argmax(sims[i])
                
                # 確定相似度等級
                if max_sim >= hi:
                    flag = 2
                    # 記錄高相似度配對
                    if names:
                        pair_info = f"{names[i]} ↔ {names[max_sim_idx]} (相似度: {max_sim:.3f})"
                    else:
                        pair_info = f"學生 {i+1} ↔ 學生 {max_sim_idx+1} (相似度: {max_sim:.3f})"
                    high_similarity_pairs.append(pair_info)
                elif max_sim >= mid:
                    flag = 1
                else:
                    flag = 0
                    
                similarity_flags.append(flag)
            
            # 記錄高相似度配對到日誌
            if high_similarity_pairs:
                logging.warning(f"檢測到 {len(high_similarity_pairs)} 對高相似度答案:")
                for pair in high_similarity_pairs:
                    logging.warning(f"  🚨 {pair}")
            
            # 準備詳細結果
            detailed_analysis = {
                "status": "completed",
                "method": "genai",
                "similarity_matrix": sims.tolist(),
                "high_similarity_pairs": high_similarity_pairs,
                "total_comparisons": len(texts) * (len(texts) - 1) // 2,
                "flagged_pairs": len(high_similarity_pairs)
            }
            
            return similarity_flags, detailed_analysis
            
        except Exception as e:
            logging.error(f"GenAI 相似度計算時發生錯誤: {e}")
            # 如果GenAI失敗，fallback到非GenAI方法
            use_api = False
    
    if not use_api:
        # 使用非 GenAI 方法（從 alternative_similarity_methods.py 導入）
        try:
            from .alternative_similarity_methods import calculate_text_similarity_enhanced, calculate_tfidf_similarity
            
            # 使用增強版本地方法：多算法融合檢測
            enhanced_results, enhanced_details = calculate_text_similarity_enhanced(texts, names, threshold=mid)
            
            # 使用TF-IDF相似度作為補充驗證
            tfidf_results, tfidf_details = calculate_tfidf_similarity(texts, threshold=mid)
            
            # 綜合兩種方法的結果（以增強方法為主）
            combined_flags = []
            for i in range(len(texts)):
                enhanced_flag = enhanced_results[i] if i < len(enhanced_results) else 0
                tfidf_flag = tfidf_results[i] if i < len(tfidf_results) else 0
                
                # 取較高的相似度標記（以增強方法為主）
                combined_flag = max(enhanced_flag, tfidf_flag)
                combined_flags.append(combined_flag)
            
            # 準備詳細結果
            detailed_analysis = {
                "status": "completed",
                "method": "enhanced_local",
                "enhanced_results": {
                    "flags": enhanced_results,
                    "details": enhanced_details
                },
                "tfidf_results": {
                    "flags": tfidf_results,
                    "details": tfidf_details
                },
                "combined_method": "max",
                "total_comparisons": len(texts) * (len(texts) - 1) // 2,
                "flagged_pairs": sum(1 for flag in combined_flags if flag >= 1)
            }
            
            return combined_flags, detailed_analysis
            
        except Exception as e:
            logging.error(f"本地相似度計算時發生錯誤: {e}")
            return [-1] * len(texts), {"status": "error", "method": "local", "error": str(e)}
    
    return [-1] * len(texts), {"status": "error", "method": "unknown"}

def get_detailed_similarity_analysis(texts: List[str], names: List[str] = None, threshold: float = 0.70) -> dict:
    """
    取得詳細的相似度分析報告
    
    Args:
        texts: 學生答案列表
        names: 學生姓名列表（可選）
        threshold: 相似度閾值
    
    Returns:
        dict: 包含相似度矩陣和詳細分析的字典
    """
    if not DO_SIMILARITY_CHECK or len(texts) < 2:
        return {"status": "skipped", "reason": "相似度檢查已關閉或文本數量不足"}
    
    try:
        # 預處理文本（與上面的函數保持一致）
        safe_texts = [t.strip() if t and t.strip() else " " for t in texts]
        
        # 計算嵌入向量
        result = genai.embed_content(
            model="models/text-embedding-004", 
            content=safe_texts, 
            task_type="RETRIEVAL_DOCUMENT"
        )
        embs = result['embedding']
        
        # 計算相似度矩陣
        sims = cosine_similarity(embs)
        np.fill_diagonal(sims, 0)
        
        # 找出高相似度配對
        high_similarity_pairs = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if sims[i][j] >= threshold:
                    student_i = names[i] if names else f"學生 {i+1}"
                    student_j = names[j] if names else f"學生 {j+1}"
                    high_similarity_pairs.append({
                        "student_1": student_i,
                        "student_2": student_j,
                        "similarity": float(sims[i][j]),
                        "index_1": i,
                        "index_2": j
                    })
        
        # 按相似度排序
        high_similarity_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "status": "completed",
            "similarity_matrix": sims.tolist(),
            "high_similarity_pairs": high_similarity_pairs,
            "total_comparisons": len(texts) * (len(texts) - 1) // 2,
            "flagged_pairs": len(high_similarity_pairs)
        }
        
    except Exception as e:
        logging.error(f"詳細相似度分析時發生錯誤: {e}")
        return {"status": "error", "error": str(e)}
