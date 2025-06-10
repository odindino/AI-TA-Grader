# analyzer.py
# 核心分析邏輯模組

import os, re, json, asyncio, logging
import pandas as pd 
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# --- Gemini API 設定 ---
def configure_gemini(api_key):
    """設定 Gemini API 金鑰並初始化模型"""
    global gmodel
    try:
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel("gemini-1.5-pro-latest")
        return True
    except Exception as e:
        logging.error(f"設定 Gemini API 金鑰時發生錯誤: {e}")
        return False

gmodel = None

# --- 使用者可修改參數 ---
TARGET_SCORE_Q = [1, 2, 3, 4, 11]
BATCH_SIZE = 6
DO_SIMILARITY_CHECK = True

# --- 評分標準 (Rubrics) ---
# 為節省空間，此處省略詳細內容，請將您先前版本的完整 RUBRICS 字典貼到此處
RUBRICS = {
    1: '''Rubric (10 pts total)\nSection A – CZ vs FZ single‑crystal Si comparison (0‑4 pts)...''',
    2: '''Rubric (10 pts total) — Student **answers any TWO** sub‑problems...''',
    3: '''Rubric (10 pts total) — Student **answers any TWO** sub‑problems...''',
    4: '''Rubric (10 pts total) — Student **answers any TWO** sub‑problems...''',
    11: '''Rubric (10 pts total)\nPart 1  Quantum confinement in 0‑D nanomaterials (0‑5)...''',
}

# --- Prompt 模板 ---
PROMPT_TEMPLATE = """You are a meticulous graduate-level TA.
First, judge whether the following answer exhibits obvious LLM writing style (including but not limited to overly smooth transitions, template phrases like 'moreover', almost no typos).
Return an integer 'ai_risk' from 0‑100 (higher = more AI-like).
{grade_block}
Respond ONLY with a valid JSON object like {{"ai_risk": ..., "score": ...}}.

Question:
{question}

Answer:
{answer}
"""

def log_to_frontend(message, callback):
    """安全地呼叫前端日誌回呼函式"""
    if callback:
        callback(message)
    else:
        print(message)

def load_exam(csv_path, log_callback):
    """從 CSV 檔案載入考卷答案"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        log_to_frontend(f"❌ 找不到指定的檔案: {csv_path}", log_callback)
        return None, None
    ans_cols = [c for c in df.columns if re.match(r'^(352|362)\d+:', str(c))]
    q_map = {i+1: col for i, col in enumerate(ans_cols)}
    log_to_frontend(f"✅ 成功載入 {os.path.basename(csv_path)}，找到 {len(df)} 位學生與 {len(q_map)} 題問答。", log_callback)
    return df, q_map

def calculate_similarity_flags(texts: list[str], hi=0.85, mid=0.70) -> list[int]:
    """計算答案間的語意相似度"""
    if not DO_SIMILARITY_CHECK or len(texts) < 2:
        return [0] * len(texts)
    try:
        safe_texts = [t if t and t.strip() else " " for t in texts]
        result = genai.embed_content(model="models/text-embedding-004", content=safe_texts, task_type="RETRIEVAL_DOCUMENT")
        embs = result['embedding']
        sims = cosine_similarity(embs)
        np.fill_diagonal(sims, 0)
        max_sims = np.max(sims, axis=1)
        return [2 if s >= hi else 1 if s >= mid else 0 for s in max_sims]
    except Exception as e:
        logging.error(f"執行相似度計算時發生錯誤: {e}")
        return [-1] * len(texts)

async def gemini_eval(question: str, rubric: str, answer: str, need_score: bool) -> tuple:
    """非同步呼叫 Gemini API 進行 AI 風格分析與評分"""
    if not answer or not answer.strip():
        return (0, 0 if need_score else None)
    grade_block = f"Then grade the answer..." if need_score else "Grading is not required..."
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer, grade_block=grade_block)
    try:
        resp = await gmodel.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json"),
            request_options={"timeout": 120}
        )
        if not resp.text: return (None, None)
        js = json.loads(resp.text)
        return (js.get("ai_risk"), js.get("score"))
    except Exception as e:
        logging.error(f"Gemini API 呼叫時發生錯誤: {e}")
        return (None, None)

async def process_question(df, qid, col, log_callback):
    """處理單一問題的所有學生答案"""
    log_to_frontend(f"➡️ 開始處理 Q{qid}...", log_callback)
    q_text = col.split(":", 1)[1].strip()
    need_score = qid in TARGET_SCORE_Q
    rubric = RUBRICS.get(qid, "")
    
    sub = df[["name", col]].rename(columns={col: "answer"}).fillna("")
    sub[f"Q{qid}_sim_flag"] = calculate_similarity_flags(sub["answer"].tolist())
    
    out_ai, out_sc = [], []
    chunks = [sub.iloc[i:i + BATCH_SIZE] for i in range(0, len(sub), BATCH_SIZE)]
    
    for i, chunk in enumerate(chunks):
        log_to_frontend(f"    Q{qid}: 處理批次 {i+1}/{len(chunks)}", log_callback)
        tasks = [gemini_eval(q_text, rubric, a, need_score) for a in chunk["answer"]]
        results = await asyncio.gather(*tasks)
        out_ai.extend([r[0] for r in results])
        out_sc.extend([r[1] for r in results])
        
    sub[f"Q{qid}_ai_risk"] = out_ai
    if need_score:
        sub[f"Q{qid}_score"] = out_sc
        
    return sub.drop(columns=["answer"])

async def run_analysis(api_key: str, csv_path: str, out_path: str, log_callback):
    """執行完整分析流程的主函式"""
    if not configure_gemini(api_key):
        log_to_frontend("❌ API 金鑰設定失敗，請檢查金鑰是否正確。", log_callback)
        return

    df, qmap = load_exam(csv_path, log_callback)
    if df is None: return

    merged_df = df[["name"]].copy()
    
    for qid, col in qmap.items():
        res_df = await process_question(df, qid, col, log_callback)
        merged_df = merged_df.merge(res_df, on="name", how="left")
        
    try:
        merged_df.to_excel(out_path, index=False, engine='openpyxl')
        log_to_frontend(f"🎉 分析完成！報告已儲存至：\n{out_path}")
    except Exception as e:
        log_to_frontend(f"❌ 儲存 Excel 報告失敗: {e}", log_callback)