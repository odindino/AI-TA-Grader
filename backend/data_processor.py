"""
數據處理模組 - 負責CSV檔案載入和學生數據處理
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Callable, Dict, List, Any


class DataProcessor:
    """數據處理器類別"""
    
    def __init__(self):
        """初始化數據處理器"""
        self.logger = logging.getLogger(__name__)
    
    def load_csv_file(self, file_path: str) -> pd.DataFrame:
        """載入CSV檔案
        
        Args:
            file_path: CSV檔案路徑
            
        Returns:
            pd.DataFrame: 載入的數據框
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"成功載入CSV檔案: {os.path.basename(file_path)}")
            return df
        except Exception as e:
            self.logger.error(f"載入CSV檔案失敗: {e}")
            raise
    
    def preprocess_responses(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """預處理學生回答數據
        
        Args:
            df: 原始數據框
            column: 要處理的欄位名稱
            
        Returns:
            Dict: 包含處理後文本和姓名的字典
        """
        try:
            # 提取文本數據
            texts = []
            names = []
            
            if column in df.columns:
                for idx, row in df.iterrows():
                    text = str(row.get(column, '')).strip()
                    texts.append(text)
                    
                    # 嘗試提取學生姓名
                    name = self._extract_student_name(row, idx)
                    names.append(name)
            
            return {
                'texts': texts,
                'names': names,
                'original_data': df
            }
            
        except Exception as e:
            self.logger.error(f"預處理回答數據失敗: {e}")
            raise
    
    def _extract_student_name(self, row: pd.Series, idx: int) -> str:
        """提取學生姓名"""
        # 嘗試不同的姓名欄位
        name_columns = ['姓名', 'Name', 'Student', 'student', '學生姓名', 'name']
        
        for col in name_columns:
            if col in row.index and pd.notna(row[col]):
                return str(row[col]).strip()
        
        # 如果找不到姓名，使用序號
        return f"學生{idx + 1}"
    
    def clean_text(self, text: str) -> str:
        """清理文本數據"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # 移除多餘的空白
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """驗證數據完整性"""
        try:
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"缺少必要欄位: {missing_columns}")
                return False
            
            # 檢查數據是否為空
            if df.empty:
                self.logger.warning("數據框為空")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"數據驗證失敗: {e}")
            return False


# 保持向後相容的函數接口
def log_to_frontend(message: str, callback: Callable):
    """安全地呼叫前端日誌回呼函式"""
    if callback:
        try:
            callback(message)
        except Exception as e:
            logging.error(f"日誌回呼函式發生錯誤: {e}")

def load_exam(csv_path: str, log_callback: Callable) -> Tuple[Optional[pd.DataFrame], dict]:
    """從 CSV 檔案載入考卷答案 - 向後相容函數"""
    """從 CSV 檔案載入考卷答案"""
    log_to_frontend(f"📊 原始檔案載入: {os.path.basename(csv_path)}", log_callback)
    
    try:
        # 載入 CSV 檔案，使用不同的編碼方式
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    csv_path, 
                    encoding=encoding,
                    dtype=str,
                    quotechar='"',
                    escapechar='\\'
                )
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            log_to_frontend("❌ 無法讀取 CSV 檔案，請檢查檔案編碼", log_callback)
            return None, {}
        
        # 清理欄位名稱
        df.columns = df.columns.str.strip().str.replace('\n', ' ', regex=False)
        
        log_to_frontend(f"📊 原始檔案載入: {len(df)} 行", log_callback)
        
        # 去重處理 - 保留每位學生分數最高的記錄
        log_to_frontend("📊 保留策略: 每位學生保留最高分數的記錄", log_callback)
        
        if 'name' not in df.columns:
            log_to_frontend("❌ 找不到 'name' 欄位", log_callback)
            return None, {}
        
        # 如果有 score 欄位，按分數排序；否則保留第一筆
        if 'score' in df.columns:
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            df = df.sort_values(['name', 'score'], ascending=[True, False])
        
        # 去除重複的學生，保留第一筆（最高分）
        df_cleaned = df.drop_duplicates(subset=['name'], keep='first')
        
        # 移除空白姓名的記錄
        df_cleaned = df_cleaned.dropna(subset=['name'])
        df_cleaned = df_cleaned[df_cleaned['name'].str.strip() != '']
        
        # 找出問題欄位 (格式: "題號: 題目內容")
        q_map = {}
        for col in df_cleaned.columns:
            if ':' in col and col != 'name':
                try:
                    # 提取題號
                    prefix = col.split(':', 1)[0].strip()
                    if prefix.isdigit():
                        qid = int(prefix)
                        q_map[qid] = col
                        log_to_frontend(f"  找到題目 {qid}: {col[:50]}...", log_callback)
                except:
                    continue
        
        if not q_map:
            log_to_frontend("❌ 未找到有效的問題欄位（格式應為 '題號: 題目內容'）", log_callback)
            return None, {}
        
        log_to_frontend(f"✅ 成功載入 {os.path.basename(csv_path)}，找到 {len(df_cleaned)} 位學生與 {len(q_map)} 題問答。", log_callback)
        return df_cleaned, q_map
        
    except Exception as e:
        log_to_frontend(f"❌ 載入檔案時發生錯誤: {e}", log_callback)
        return None, {}
