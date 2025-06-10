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
    
    async def grade_responses_batch(self, texts: List[str], rubric: str, question_id: int) -> tuple:
        """批次評分學生回答
        
        Args:
            texts: 學生回答文本列表
            rubric: 評分標準
            question_id: 問題編號
            
        Returns:
            tuple: (分數列表, AI風險列表)
        """
        if not self.model:
            raise ValueError("Gemini模型未初始化")
        
        scores = []
        ai_risks = []
        
        for text in texts:
            try:
                score, ai_risk = await self._grade_single_response(text, rubric, question_id)
                scores.append(score)
                ai_risks.append(ai_risk)
            except Exception as e:
                self.logger.error(f"評分單個回答失敗: {e}")
                scores.append(0.0)
                ai_risks.append(0)
        
        return scores, ai_risks
    
    async def _grade_single_response(self, text: str, rubric: str, question_id: int) -> tuple:
        """評分單個學生回答，同時返回分數和AI風險"""
        try:
            # 構建正確的 prompt
            grade_block = f"Then score this answer based on the rubric, returning an integer from 0‑10.\n\nRubric:\n{rubric}"
            question = f"Question {question_id}"
            
            prompt = PROMPT_TEMPLATE.format(
                grade_block=grade_block,
                question=question,
                answer=text
            )
            
            self.logger.info(f"🤖 發送給 Gemini 的 Prompt (Q{question_id}):")
            self.logger.info(f"📝 學生回答: {text[:100]}...")
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.1,
                    candidate_count=1
                )
            )
            
            # 檢查回應狀態
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "NO_CANDIDATES"
                self.logger.warning(f"🚨 API回應被過濾 (finish_reason: {finish_reason})，使用本地評分")
                # 返回本地評分結果
                local_score = self._estimate_local_score(text)
                return local_score, 0  # AI風險設為0（因為API過濾不能判斷）
            
            # 記錄 AI 回應
            response_text = response.text.strip()
            self.logger.info(f"🤖 Gemini 回應 (Q{question_id}): {response_text}")
            
            # 解析回應中的分數和AI風險
            score, ai_risk = self._extract_score_and_ai_risk_from_response(response_text)
            self.logger.info(f"📊 解析出的分數 (Q{question_id}): {score}")
            self.logger.info(f"🔍 解析出的AI風險 (Q{question_id}): {ai_risk}")
            
            return score, ai_risk
            
        except Exception as e:
            self.logger.error(f"評分失敗: {e}")
            # 使用本地評分作為後備
            local_score = self._estimate_local_score(text)
            self.logger.info(f"💻 使用本地評分: {local_score}")
            return local_score, 0
    
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
    
    def _extract_score_and_ai_risk_from_response(self, response_text: str) -> tuple:
        """從Gemini回應中提取分數和AI風險"""
        import re
        import json
        
        score = 0.0
        ai_risk = 0
        
        try:
            # 嘗試解析JSON格式
            json_match = re.search(r'\{.*?"ai_risk".*?"score".*?\}|\{.*?"score".*?"ai_risk".*?\}', response_text, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group(0))
                ai_risk = int(json_data.get('ai_risk', 0))
                score = float(json_data.get('score', 0))
                return score, ai_risk
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 如果JSON解析失敗，分別提取數字
        ai_risk_match = re.search(r'"ai_risk":\s*(\d+)', response_text)
        if ai_risk_match:
            ai_risk = int(ai_risk_match.group(1))
        
        score_match = re.search(r'"score":\s*([0-9.]+)', response_text)
        if score_match:
            score = float(score_match.group(1))
        elif not score_match:
            # 尋找任何數字作為分數
            all_numbers = re.findall(r'\b([0-9.]+)\b', response_text)
            if all_numbers:
                score = float(all_numbers[-1])  # 取最後一個數字
                score = min(score, 10.0)  # 限制最高分為10分
        
        return score, ai_risk
    
    def _estimate_local_score(self, text: str) -> float:
        """本地評分估算（從analyzer.py複製）"""
        if not text.strip():
            return 0.0
        
        text_clean = text.strip().lower()
        
        # 基於文本長度的基礎分數
        word_count = len(text_clean.split())
        if word_count < 20:
            base_score = 2.0
        elif word_count < 50:
            base_score = 4.0
        elif word_count < 100:
            base_score = 6.0
        elif word_count < 200:
            base_score = 7.0
        else:
            base_score = 8.0
        
        # 檢查是否有結構化回答
        structure_bonus = 0.0
        if any(marker in text for marker in ['1.', '2.', '3.', '•', '-', 'advantage', 'disadvantage']):
            structure_bonus = 0.5
        
        final_score = min(base_score + structure_bonus, 10.0)
        return max(final_score, 1.0) if text.strip() else 0.0


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
