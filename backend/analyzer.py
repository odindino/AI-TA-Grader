# refactored_analyzer.py
# 核心分析邏輯模組 - 重構版本

import os, re, json, asyncio, logging
import pandas as pd 
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# 引入本地模組
from .config import TARGET_SCORE_Q, BATCH_SIZE, DO_SIMILARITY_CHECK, RUBRICS
from .gemini_client import GeminiClient
from .data_processor import DataProcessor
from .visualization import VisualizationEngine
from .similarity_detector import SimilarityDetector


class AnalysisEngine:
    """重構後的分析引擎類別"""
    
    def __init__(self, api_key: Optional[str] = None):
        """初始化分析引擎
        
        Args:
            api_key: Gemini API 金鑰（可選）
        """
        self.gemini_client = GeminiClient(api_key) if api_key else None
        self.data_processor = DataProcessor()
        self.visualization = VisualizationEngine()
        self.similarity_detector = SimilarityDetector()
        self.use_genai = api_key is not None
        
        # 設定日誌
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def configure_gemini(self, api_key: str, model_name: str = "gemini-1.5-pro-latest") -> bool:
        """設定 Gemini API
        
        Args:
            api_key: API 金鑰
            model_name: 模型名稱
            
        Returns:
            bool: 設定是否成功
        """
        try:
            self.gemini_client = GeminiClient(api_key, model_name)
            self.use_genai = True
            return True
        except Exception as e:
            self.logger.error(f"設定 Gemini API 時發生錯誤: {e}")
            return False
    
    async def process_question(self, df: pd.DataFrame, qid: int, col: str, 
                             log_callback: callable = None, use_genai: bool = None) -> pd.DataFrame:
        """處理單個問題的分析
        
        Args:
            df: 數據框
            qid: 問題編號
            col: 欄位名稱
            log_callback: 日誌回調函數
            use_genai: 是否使用GenAI（None則使用預設）
            
        Returns:
            pd.DataFrame: 處理後的數據框
        """
        if use_genai is None:
            use_genai = self.use_genai
            
        try:
            # 資料預處理
            processed_data = self.data_processor.preprocess_responses(df, col)
            
            # 相似度檢測 - 只使用本地方法
            if DO_SIMILARITY_CHECK:
                similarity_results = await self._calculate_similarity_batch(
                    processed_data['texts'], 
                    processed_data['names'],
                    use_genai=False  # 強制使用本地方法
                )
                processed_data['similarity_scores'] = similarity_results['scores']
                processed_data['similarity_matrix'] = similarity_results['matrix']
                processed_data['similarity_info'] = similarity_results['info']
            
            # 批次評分
            if use_genai and self.gemini_client:
                scores, ai_risks = await self._grade_responses_genai(processed_data, qid, log_callback)
            else:
                scores = self._grade_responses_local(processed_data, qid)
                ai_risks = [0] * len(scores)  # 本地評分沒有AI風險評估
            
            # 創建結果數據框
            result_df = df.copy()
            result_df[f'Q{qid}_分數'] = scores
            result_df[f'Q{qid}_AI風險'] = ai_risks
            
            if DO_SIMILARITY_CHECK:
                result_df[f'Q{qid}_相似度分數'] = processed_data['similarity_scores']
            
            # 生成視覺化
            if len(processed_data['texts']) > 1:
                viz_results = await self.visualization.create_similarity_matrices(
                    processed_data['texts'],
                    processed_data['names'],
                    qid,
                    use_genai=use_genai
                )
                # 添加矩陣數據到視覺化結果中
                if DO_SIMILARITY_CHECK and 'similarity_matrix' in processed_data:
                    viz_results['matrix_data'] = processed_data['similarity_matrix']
                    viz_results['names'] = processed_data['names']
                
                result_df.attrs[f'Q{qid}_visualization'] = viz_results
            
            if log_callback:
                log_callback(f"Q{qid} 處理完成")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"處理問題 Q{qid} 時發生錯誤: {e}")
            raise
    
    async def _calculate_similarity_batch(self, texts: List[str], names: List[str] = None,
                                        use_genai: bool = True) -> Dict[str, Any]:
        """批次計算相似度
        
        Args:
            texts: 文本列表
            names: 姓名列表
            use_genai: 是否使用GenAI
            
        Returns:
            Dict: 相似度結果
        """
        try:
            if use_genai and self.gemini_client:
                return await self.similarity_detector.calculate_genai_similarity(
                    texts, names, self.gemini_client
                )
            else:
                return self.similarity_detector.calculate_local_similarity(texts, names)
        except Exception as e:
            self.logger.error(f"相似度計算失敗: {e}")
            return {
                'flags': [-1] * len(texts),
                'info': {"status": "error", "method": "unknown"}
            }
    
    async def _grade_responses_genai(self, data: Dict[str, Any], qid: int, 
                                   log_callback: callable = None) -> tuple:
        """使用GenAI評分回答
        
        Args:
            data: 處理後的數據
            qid: 問題編號
            log_callback: 日誌回調
            
        Returns:
            tuple: (分數列表, AI風險列表)
        """
        if not self.gemini_client:
            raise ValueError("Gemini客戶端未初始化")
        
        texts = data['texts']
        rubric = RUBRICS.get(qid, "")
        scores = []
        ai_risks = []
        
        # 批次處理
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            
            try:
                batch_scores, batch_ai_risks = await self.gemini_client.grade_responses_batch(
                    batch_texts, rubric, qid
                )
                scores.extend(batch_scores)
                ai_risks.extend(batch_ai_risks)
                
                if log_callback:
                    log_callback(f"Q{qid} 批次 {i//BATCH_SIZE + 1} 完成")
                    
            except Exception as e:
                self.logger.error(f"批次評分失敗: {e}")
                # 使用本地評分作為後備
                fallback_scores = [self._estimate_score_local(text, qid) for text in batch_texts]
                fallback_ai_risks = [0] * len(batch_texts)  # 本地評分沒有AI風險
                scores.extend(fallback_scores)
                ai_risks.extend(fallback_ai_risks)
        
        return scores, ai_risks
    
    def _grade_responses_local(self, data: Dict[str, Any], qid: int) -> List[float]:
        """使用本地方法評分回答
        
        Args:
            data: 處理後的數據
            qid: 問題編號
            
        Returns:
            List[float]: 分數列表
        """
        texts = data['texts']
        return [self._estimate_score_local(text, qid) for text in texts]
    
    def _estimate_score_local(self, text: str, qid: int) -> float:
        """本地評分估計
        
        Args:
            text: 回答文本
            qid: 問題編號
            
        Returns:
            float: 估計分數
        """
        if not text.strip():
            return 0.0
        
        text_clean = text.strip().lower()
        
        # 基於文本長度的基礎分數（更合理的範圍）
        word_count = len(text_clean.split())
        if word_count < 20:
            base_score = 2.0  # 太短
        elif word_count < 50:
            base_score = 4.0  # 一般
        elif word_count < 100:
            base_score = 6.0  # 中等
        elif word_count < 200:
            base_score = 7.0  # 較好
        else:
            base_score = 8.0  # 很好
        
        # 問題特定的關鍵詞加分
        keywords_bonus = self._calculate_keywords_bonus(text, qid)
        
        # 檢查是否有結構化回答（列表、數字等）
        structure_bonus = 0.0
        if any(marker in text for marker in ['1.', '2.', '3.', '•', '-', 'advantage', 'disadvantage']):
            structure_bonus = 0.5
        
        final_score = min(base_score + keywords_bonus + structure_bonus, 10.0)
        
        # 確保最低分不低於1分（如果有內容的話）
        return max(final_score, 1.0) if text.strip() else 0.0
    
    def _calculate_keywords_bonus(self, text: str, qid: int) -> float:
        """計算關鍵詞加分
        
        Args:
            text: 文本
            qid: 問題編號
            
        Returns:
            float: 加分
        """
        text_lower = text.lower()
        bonus = 0.0
        
        # 問題特定關鍵詞
        question_keywords = {
            1: ['cz', 'fz', 'czochralski', 'float zone', 'channeling', 'crystal', 'silicon'],
            2: ['cvd', 'mbe', 'epitaxy', 'molecular beam', 'chemical vapor'],
            3: ['gaas', 'silicon', 'bandgap', 'heterojunction', 'solar cell'],
            4: ['mosfet', 'transistor', 'threshold', 'channel'],
            11: ['process', 'fabrication', 'semiconductor']
        }
        
        keywords = question_keywords.get(qid, [])
        for keyword in keywords:
            if keyword in text_lower:
                bonus += 0.5
        
        return min(bonus, 3.0)  # 最多3分加分
    
    def _find_question_columns(self, df: pd.DataFrame, target_questions: List[int] = None) -> Dict[int, str]:
        """自動檢測所有問題欄位
        
        Args:
            df: 數據框
            target_questions: 目標問題編號列表（已廢棄，改為自動檢測）
            
        Returns:
            Dict[int, str]: 問題編號到欄位名稱的映射
        """
        question_cols = {}
        
        # 自動檢測所有問題欄位（以數字開頭或包含常見問題ID模式）
        potential_question_cols = []
        for col in df.columns:
            col_str = str(col).strip()
            # 檢測包含問題ID的欄位（352xxx, 362xxx等Canvas格式）
            if (col_str and 
                (col_str.startswith('352') or 
                 col_str.startswith('362') or 
                 col_str.startswith('Q') or
                 any(char.isdigit() for char in col_str[:10])) and
                '10.0' not in col_str and  # 排除分數欄位
                'pts' not in col_str.lower() and
                len(col_str) > 10):  # 問題欄位通常比較長
                potential_question_cols.append(col)
        
        # 為檢測到的問題欄位分配序號
        for i, col in enumerate(potential_question_cols, 1):
            question_cols[i] = col
            
        # 記錄檢測結果
        if hasattr(self, 'logger'):
            self.logger.info(f"自動檢測到 {len(question_cols)} 個問題欄位")
            for qid, col in question_cols.items():
                col_preview = col[:80] + '...' if len(col) > 80 else col
                self.logger.info(f"  Q{qid}: {col_preview}")
        
        return question_cols
    
    async def analyze_complete_dataset(self, file_path: str, 
                                     log_callback: callable = None) -> Dict[str, Any]:
        """分析完整數據集
        
        Args:
            file_path: CSV檔案路徑
            log_callback: 日誌回調函數
            
        Returns:
            Dict: 分析結果
        """
        try:
            # 載入數據
            df = self.data_processor.load_csv_file(file_path)
            if log_callback:
                log_callback(f"載入數據: {len(df)} 筆記錄")
            
            results = {}
            all_visualizations = {}
            
            # 自動檢測所有問題欄位
            question_cols = self._find_question_columns(df)
            
            if log_callback:
                log_callback(f"檢測到 {len(question_cols)} 個問題欄位，將逐一處理...")
                for qid, col in list(question_cols.items())[:3]:  # 顯示前3個作為示例
                    col_preview = col[:50] + '...' if len(col) > 50 else col
                    log_callback(f"  Q{qid}: {col_preview}")
            
            # 處理每個問題
            for qid, col in question_cols.items():
                if log_callback:
                    log_callback(f"開始處理 Q{qid} (欄位: {col[:50]}...)")
                
                result_df = await self.process_question(df, qid, col, log_callback)
                results[f'Q{qid}'] = result_df
                
                # 收集視覺化結果
                if hasattr(result_df, 'attrs') and f'Q{qid}_visualization' in result_df.attrs:
                    all_visualizations[f'Q{qid}'] = result_df.attrs[f'Q{qid}_visualization']
            
            # 合併結果
            final_df = df.copy()  # 確保final_df總是被定義
            if results:
                # 從第一個結果開始構建final_df
                first_result_key = list(results.keys())[0]
                final_df = results[first_result_key].copy()
                
                # 合併其他問題的結果
                for result_key, result_df in results.items():
                    qid = int(result_key.replace('Q', ''))
                    score_col = f'Q{qid}_分數'
                    similarity_col = f'Q{qid}_相似度分數'
                    ai_risk_col = f'Q{qid}_AI風險'
                    
                    if score_col in result_df.columns and score_col not in final_df.columns:
                        final_df[score_col] = result_df[score_col]
                    
                    if similarity_col in result_df.columns and similarity_col not in final_df.columns:
                        final_df[similarity_col] = result_df[similarity_col]
                    
                    if ai_risk_col in result_df.columns and ai_risk_col not in final_df.columns:
                        final_df[ai_risk_col] = result_df[ai_risk_col]
            
            # 生成HTML報告
            html_report = await self.visualization.generate_enhanced_html_report(
                final_df, all_visualizations
            )
            
            return {
                'dataframe': final_df,
                'visualizations': all_visualizations,
                'html_report': html_report,
                'summary': self._generate_summary(final_df)
            }
            
        except Exception as e:
            self.logger.error(f"完整數據集分析失敗: {e}")
            raise
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成分析摘要
        
        Args:
            df: 結果數據框
            
        Returns:
            Dict: 摘要信息
        """
        summary = {
            'total_students': len(df),
            'questions_analyzed': [],
            'average_scores': {},
            'similarity_detected': {}
        }
        
        for qid in TARGET_SCORE_Q:
            score_col = f'Q{qid}_分數'
            similarity_col = f'Q{qid}_相似度標記'
            
            if score_col in df.columns:
                summary['questions_analyzed'].append(qid)
                scores = pd.to_numeric(df[score_col], errors='coerce')
                summary['average_scores'][f'Q{qid}'] = float(scores.mean())
                
                if similarity_col in df.columns:
                    similarity_flags = df[similarity_col]
                    high_similarity = sum(1 for flag in similarity_flags if flag > 0)
                    summary['similarity_detected'][f'Q{qid}'] = high_similarity
        
        return summary


# 保持向後兼容的函數接口
async def process_question(df: pd.DataFrame, qid: int, col: str, 
                         log_callback: callable = None, use_genai: bool = True) -> pd.DataFrame:
    """向後兼容的問題處理函數"""
    engine = AnalysisEngine()
    return await engine.process_question(df, qid, col, log_callback, use_genai)


def configure_gemini(api_key: str, model_name: str = "gemini-1.5-pro-latest") -> bool:
    """向後兼容的Gemini設定函數"""
    engine = AnalysisEngine()
    return engine.configure_gemini(api_key, model_name)
