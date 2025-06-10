# analyzer.py
# 核心分析邏輯模組

import os, re, json, asyncio, logging
import pandas as pd 
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import yaml # 新增匯入 yaml

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
        df = pd.read_csv(
            csv_path,
            encoding='utf-8',
            dtype=str,
            quotechar='"',
            escapechar='\\'
        )
        # 去除欄位名稱前後空白，並將換行轉為空格
        df.columns = (
            df.columns
            .str.strip()
            .str.replace('\n', ' ', regex=False)
        )
    except FileNotFoundError:
        log_to_frontend(f"❌ 找不到指定的檔案: {csv_path}", log_callback)
        return None, None
    
    # 找到結構標記欄位的索引
    try:
        attempt_idx = df.columns.get_loc('attempt')
        n_correct_idx = df.columns.get_loc('n correct')
    except KeyError as e:
        log_to_frontend(f"❌ 找不到必要的欄位: {e}，請檢查CSV格式", log_callback)
        return None, None
    
    # 解析attempt和n correct之間的題目/分數配對
    q_map = {}
    q_counter = 1
    
    for i in range(attempt_idx + 1, n_correct_idx, 2):
        if i + 1 < n_correct_idx:  # 確保有配對的分數欄位
            question_col = df.columns[i]
            score_col = df.columns[i + 1]
            
            # 檢查分數欄位是否為數值型態或包含分數資訊
            if (pd.to_numeric(df[score_col], errors='coerce').notna().any() or 
                any(str(val).replace('.', '').isdigit() for val in df[score_col].dropna())):
                q_map[q_counter] = question_col
                q_counter += 1
                log_to_frontend(f"  找到題目 {q_counter-1}: {question_col[:50]}...", log_callback)
    
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

async def run_analysis(api_key: str, csv_path: str, out_base_path: str, log_callback):
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
        
    # 定義各種格式的輸出路徑
    xlsx_path = f"{out_base_path}.xlsx"
    csv_path_out = f"{out_base_path}.csv"
    yaml_path = f"{out_base_path}.yaml"
    html_path = f"{out_base_path}.html"

    try:
        # 儲存為 Excel
        merged_df.to_excel(xlsx_path, index=False, engine='openpyxl')
        log_to_frontend(f"🎉 Excel 報告已儲存至：\n{xlsx_path}", log_callback)

        # 儲存為 CSV
        merged_df.to_csv(csv_path_out, index=False)
        log_to_frontend(f"🎉 CSV 報告已儲存至：\n{csv_path_out}", log_callback)

        # 儲存為 YAML
        # 將 DataFrame 轉換為字典列表以便 YAML 序列化
        yaml_data = merged_df.to_dict(orient='records')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        log_to_frontend(f"🎉 YAML 報告已儲存至：\n{yaml_path}", log_callback)

        # 儲存為 HTML
        # 使用 DataFrame.to_html() 方法，可以加入一些樣式
        html_content = merged_df.to_html(index=False, escape=False, classes='table table-striped')
        # 可以選擇性地加入一些基本的 HTML 結構和 CSS 樣式
        html_output = f"""
        <html>
        <head>
            <title>分析報告</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .table-striped tbody tr:nth-of-type(odd) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>分析報告</h1>
            {html_content}
        </body>
        </html>
        """
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        log_to_frontend(f"🎉 HTML 報告已儲存至：\n{html_path}", log_callback)

    except Exception as e:
        log_to_frontend(f"❌ 儲存報告失敗: {e}", log_callback)