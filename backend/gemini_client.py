"""
Gemini API 客戶端模組 - 處理與 Google Gemini API 的所有交互
"""

import json
import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple
import google.generativeai as genai

from .config import PROMPT_TEMPLATE


class GeminiClient:
    """Gemini API 客戶端類別"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro-latest"):
        """初始化Gemini客戶端
        
        Args:
            api_key: Gemini API 金鑰
            model_name: 使用的模型名稱
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        if api_key:
            self._configure_gemini(api_key, model_name)
    
    def _configure_gemini(self, api_key: str, model_name: str) -> bool:
        """設定 Gemini API 金鑰並初始化模型"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            return True
        except Exception as e:
            self.logger.error(f"設定 Gemini API 金鑰時發生錯誤: {e}")
            return False
    
    async def grade_responses_batch(self, texts: List[str], rubric: str, question_id: int) -> List[float]:
        """批次評分學生回答
        
        Args:
            texts: 學生回答文本列表
            rubric: 評分標準
            question_id: 問題編號
            
        Returns:
            List[float]: 分數列表
        """
        if not self.model:
            raise ValueError("Gemini模型未初始化")
        
        scores = []
        
        for text in texts:
            try:
                score = await self._grade_single_response(text, rubric, question_id)
                scores.append(score)
            except Exception as e:
                self.logger.error(f"評分單個回答失敗: {e}")
                scores.append(0.0)
        
        return scores
    
    async def _grade_single_response(self, text: str, rubric: str, question_id: int) -> float:
        """評分單個學生回答"""
        try:
            prompt = PROMPT_TEMPLATE.format(rubric=rubric, answer=text)
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.1
                )
            )
            
            # 解析回應中的分數
            score_text = response.text.strip()
            score = self._extract_score_from_response(score_text)
            
            return score
            
        except Exception as e:
            self.logger.error(f"評分失敗: {e}")
            return 0.0
    
    def _extract_score_from_response(self, response_text: str) -> float:
        """從Gemini回應中提取分數"""
        import re
        
        # 尋找 JSON 格式的分數
        json_match = re.search(r'\{.*?"score":\s*([0-9.]+).*?\}', response_text, re.DOTALL)
        if json_match:
            try:
                return float(json_match.group(1))
            except ValueError:
                pass
        
        # 尋找數字格式的分數
        number_matches = re.findall(r'(?:分數|score|得分).*?([0-9.]+)', response_text, re.IGNORECASE)
        if number_matches:
            try:
                return float(number_matches[0])
            except ValueError:
                pass
        
        # 尋找任何數字
        all_numbers = re.findall(r'\b([0-9.]+)\b', response_text)
        if all_numbers:
            try:
                score = float(all_numbers[0])
                return min(score, 10.0)  # 限制最高分為10分
            except ValueError:
                pass
        
        return 0.0


# 保持向後相容的函數接口
def configure_gemini(api_key: str, model_name: str = "gemini-1.5-pro-latest") -> bool:
    """向後相容的配置函數"""
    try:
        client = GeminiClient(api_key, model_name)
        return client.model is not None
    except Exception:
        return False
        return True
    except Exception as e:
        logging.error(f"設定 Gemini API 金鑰時發生錯誤: {e}")
        return False

async def gemini_eval(question: str, rubric: str, answer: str, need_score: bool) -> Tuple[int, float]:
    """非同步呼叫 Gemini API 進行 AI 風格分析與評分"""
    global gmodel
    
    if gmodel is None:
        raise ValueError("Gemini 模型尚未初始化")
    
    if need_score:
        grade_block = f"Then score this answer based on the rubric, returning an integer from 0‑10.\n\nRubric:\n{rubric}"
    else:
        grade_block = "Do NOT provide a score, just return 'score': null."
        
    prompt = PROMPT_TEMPLATE.format(
        grade_block=grade_block,
        question=question,
        answer=answer
    )
    
    try:
        response = await gmodel.generate_content_async(prompt)
        
        if not response.text:
            logging.warning("Gemini API 回應為空")
            return 0, None if not need_score else 0
            
        # 使用正則表達式提取 JSON
        import re
        json_pattern = r'\{[^{}]*"ai_risk"\s*:\s*\d+[^{}]*\}'
        matches = re.findall(json_pattern, response.text)
        
        if matches:
            json_str = matches[0]
        else:
            # 如果找不到完整的 JSON，嘗試解析整個回應
            json_str = response.text.strip()
            
        try:
            result = json.loads(json_str)
            ai_risk = result.get("ai_risk", 0)
            score = result.get("score") if need_score else None
            return ai_risk, score
        except json.JSONDecodeError:
            # 如果 JSON 解析失敗，嘗試提取數字
            ai_risk_match = re.search(r'"ai_risk"\s*:\s*(\d+)', response.text)
            score_match = re.search(r'"score"\s*:\s*(\d+)', response.text) if need_score else None
            
            ai_risk = int(ai_risk_match.group(1)) if ai_risk_match else 0
            score = int(score_match.group(1)) if score_match and need_score else None
            
            return ai_risk, score
            
    except Exception as e:
        logging.error(f"Gemini API 呼叫失敗: {e}")
        return 0, None if not need_score else 0
