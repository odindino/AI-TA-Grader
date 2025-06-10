# analyzer.py
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

# --- 評分標準 (Rubrics) ---
RUBRICS = {
1: '''Rubric (10 pts total)

Section A – CZ vs FZ single‑crystal Si comparison (0‑4 pts)
  • Clearly list ≥ 2 advantages and ≥ 2 disadvantages for **each** process  
    – e.g. CZ: large diameter, cheaper / higher O contamination, lower resistivity  
    – FZ: crucible‑less, ultra‑high purity, high resistivity / smaller boule, costly  
  • 0.5 pt per correct advantage/disadvantage, up to 4 pts

Section B – Channeling effect definition & impact (0‑3 pts)
  • Defines channeling as ions travelling along low‑index crystal axes/planes (1 pt)  
  • Describes deeper projected range / dose loss / tail in dopant profile (1 pt)  
  • Mentions dependence on crystal orientation/energy (1 pt)

Section C – Mitigation methods (0‑3 pts)
  Any three of the following, 1 pt each (max 3 pts):  
    – Tilt/rotate wafer during implantation  
    – Use amorphizing pre‑implant (e.g. Si, Ge)  
    – Grow/retain surface oxide or SiN mask  
    – Implant through amorphous layer (screen oxide)  
    – Use random beam incidence or beam wobbling
''',
    2: '''Rubric (10 pts total) — Student **answers any TWO** sub‑problems  
Score each answered sub‑problem 0‑5 pts, keep the best two (max 10)

Sub‑problem 1  Mass‑transport‑ vs. surface‑reaction‑limited CVD (0‑5)
  • Correct definition of each regime & rate‑determining step (2 pts)  
  • Describes dependence on temperature, pressure, boundary layer (1 pt)  
  • Mentions impact on thickness uniformity or step coverage (1 pt)  
  • Gives practical example or sketch of concentration profile (1 pt)

Sub‑problem 2  MBE working principle (0‑5)
  • UHV environment & effusion cells produce atom/molecule beams (2 pts)  
  • Ballistic arrival / adsorption–surface diffusion–incorporation process (1 pt)  
  • In‑situ monitoring (e.g. RHEED) & precise flux control (1 pt)  
  • Typical growth rate (~1 µm h⁻¹) & ultra‑high purity advantage (1 pt)

Sub‑problem 3  Exceeding critical thickness in heteroepitaxy (0‑5)
  • Introduces misfit strain & Matthews–Blakeslee criterion (1 pt)  
  • Formation of misfit dislocations / strain relaxation (2 pts)  
  • Possible 3‑D islanding (S–K), surface roughening or cracks (1 pt)  
  • Electrical/optical degradation consequence (1 pt)
''',
    3: '''Rubric (10 pts total) — Student **answers any TWO** sub‑problems  
Score each answered sub‑problem 0‑5 pts, keep the best two (max 10)

Sub‑problem 1  Si vs GaAs band structure differences (0‑5)
  • Indirect (Si) vs direct (GaAs) bandgap nature & values (2 pts)  
  • Conduction‑band valley positions / density‑of‑states (1 pt)  
  • Carrier mobility & effective mass comparison (1 pt)  
  • Consequence for optoelectronic efficiency (1 pt)

Sub‑problem 2  “Straddling gap” heterojunction (Type‑I) (0‑5)
  • Correctly picks Type‑I (1 pt)  
  • Conduction & valence bands of wider‑gap material both higher/lower enclosing narrow‑gap (2 pts)  
  • Draws / verbally describes band diagram & carrier confinement (2 pts)

Sub‑problem 3  Multijunction solar‑cell spectral utilization (0‑5)
  • Bandgap stacking / current matching concept (2 pts)  
  • Use of tunnel junctions / graded buffers (1 pt)  
  • Spectrum splitting or lattice‑matched material selection (1 pt)  
  • Anti‑reflection or light‑management strategy (1 pt)
''',
    4: '''Rubric (10 pts total) — Student **answers any TWO** sub‑problems  
Score each answered sub‑problem 0‑5 pts, keep the best two (max 10)

Sub‑problem 1  Early challenges in GaN on sapphire (0‑5)
  • ~16 % lattice mismatch & thermal‑expansion mismatch (2 pts)  
  • High threading dislocation density / cracking (1 pt)  
  • Poor surface wetting, nucleation issues prior to LT‑buffer invention (1 pt)  
  • Impact on LED efficiency / reliability (1 pt)

Sub‑problem 2  Difficulty of achieving p‑type GaN (0‑5)
  • Deep acceptor level of Mg (≈200 meV) limits hole activation (2 pts)  
  • H passivation forming Mg–H complexes, need anneal (1 pt)  
  • Compensation by native donors / defects (1 pt)  
  • Historically low hole mobility / conductivity (1 pt)

Sub‑problem 3  (D) Lower manufacturing cost? — critical discussion (0‑5)
  • States that early GaN/Sapphire actually ↑ cost due to low yield (1 pt)  
  • Explains how subsequent mass‑production & cheap sapphire made cost reasonable (1 pt)  
  • Compares to SiC, bulk GaN or phosphor‑converted alternatives (1 pt)  
  • Evaluates epi reactor throughput, wafer price, device efficiency vs cost (2 pts)
''',
    11: '''Rubric (10 pts total)

Part 1  Quantum confinement in 0‑D nanomaterials (0‑5)
  • Defines confinement when particle size ≲ exciton Bohr radius (1 pt)  
  • Energy levels become discrete, bandgap widens with decreasing size (2 pts)  
  • Mentions size‑tunable optical emission / quantum‑size effect (1 pt)  
  • Gives formula E ∝ 1/L² or cites typical CdSe Q‑dot example (1 pt)

Part 2  Si nanowires as Li‑ion anode (0‑5)
  • Lists Si theoretical capacity ≈4200 mAh g⁻¹ (1 pt)  
  • bulk Si issues: >300 % volume expansion, pulverization, loss of contact (2 pts)  
  • SiNWs accommodate strain radially / maintain electrical pathway (1 pt)  
  • Large surface area for fast Li diffusion & facile SEI formation control (1 pt)
''',
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
        
        log_to_frontend(f"📊 原始檔案載入: {len(df)} 行", log_callback)
        
        # 檢查是否有 'name' 欄位
        if 'name' not in df.columns:
            log_to_frontend("❌ 找不到 'name' 欄位，請檢查CSV格式", log_callback)
            return None, None
            
        # 去除重複的學生記錄，優先保留最高分數的記錄
        original_count = len(df)
        
        # 檢查是否有 score 欄位來判斷最高分
        if 'score' in df.columns:
            # 將分數轉換為數值型態
            df['score_numeric'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
            # 根據姓名分組，保留每組中分數最高的記錄
            df = df.loc[df.groupby('name')['score_numeric'].idxmax()]
            df = df.drop(columns=['score_numeric'])  # 移除臨時欄位
            log_to_frontend("📊 保留策略: 每位學生保留最高分數的記錄", log_callback)
        else:
            # 如果沒有分數欄位，則保留第一次出現的記錄
            df = df.drop_duplicates(subset=['name'], keep='first')
            log_to_frontend("📊 保留策略: 每位學生保留第一次記錄", log_callback)
        
        deduplicated_count = len(df)
        
        if original_count != deduplicated_count:
            removed_count = original_count - deduplicated_count
            log_to_frontend(f"🔄 去除重複記錄: {removed_count} 個重複作答，保留 {deduplicated_count} 位學生", log_callback)
        
        # 去除姓名為空的行
        df = df.dropna(subset=['name'])
        df = df[df['name'].str.strip() != '']
        final_count = len(df)
        
        if deduplicated_count != final_count:
            log_to_frontend(f"🧹 已移除空白姓名: {deduplicated_count} -> {final_count} 行", log_callback)
            
    except FileNotFoundError:
        log_to_frontend(f"❌ 找不到指定的檔案: {csv_path}", log_callback)
        return None, None
    except Exception as e:
        log_to_frontend(f"❌ 讀取CSV檔案時發生錯誤: {e}", log_callback)
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

def calculate_similarity_flags(texts: list[str], names: list[str] = None, hi=0.85, mid=0.70, min_length=50, use_api=True) -> tuple:
    """
    計算答案間的相似度，檢測潛在抄襲
    支援 GenAI (需要API) 和非GenAI方法 (本地計算)
    誒
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
    
    
    if use_api and gmodel:
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
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
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
            from alternative_similarity_methods import calculate_text_similarity_enhanced, calculate_tfidf_similarity
            
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

def get_detailed_similarity_analysis(texts: list[str], names: list[str] = None, threshold=0.70):
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

async def gemini_eval(question: str, rubric: str, answer: str, need_score: bool) -> tuple:
    """非同步呼叫 Gemini API 進行 AI 風格分析與評分"""
    if not answer or not answer.strip():
        return (0, 0 if need_score else None)
    
    if need_score:
        grade_block = f"""Then grade the answer based on this rubric (return an integer score 0-10):

{rubric}

Grade the answer objectively according to the rubric criteria."""
    else:
        grade_block = "Grading is not required for this question."
    
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

async def process_question(df, qid, col, log_callback, use_genai=True):
    """處理單一問題的所有學生答案"""
    log_to_frontend(f"➡️ 開始處理 Q{qid}...", log_callback)
    q_text = col.split(":", 1)[1].strip()
    need_score = qid in TARGET_SCORE_Q
    rubric = RUBRICS.get(qid, "")
    
    sub = df[["name", col]].rename(columns={col: "answer"}).fillna("")
    
    # 相似度檢查（同時使用GenAI和非GenAI方法，根據use_genai決定）
    sim_flags, sim_details = calculate_similarity_flags(
        sub["answer"].tolist(), 
        names=sub["name"].tolist(),
        use_api=use_genai
    )
    sub[f"Q{qid}_sim_flag"] = sim_flags
    
    if use_genai:
        # 使用GenAI進行AI風險分析和評分
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
    else:
        # 不使用GenAI，設置預設值
        sub[f"Q{qid}_ai_risk"] = [0] * len(sub)  # 不進行AI風險分析
        if need_score:
            sub[f"Q{qid}_score"] = [None] * len(sub)  # 不進行評分
        log_to_frontend(f"    Q{qid}: 跳過AI分析和評分（無API金鑰）", log_callback)
    
    # 創建相似度矩陣視覺化
    log_to_frontend(f"    Q{qid}: 創建相似度分析視覺化...", log_callback)
    genai_matrix, local_matrix = create_similarity_matrix_visualization(
        sub["answer"].tolist(), 
        sub["name"].tolist(), 
        str(qid)
    )
    
    # 將視覺化結果添加到結果中
    if genai_matrix:
        sub[f"Q{qid}_genai_matrix"] = [genai_matrix] * len(sub)
    if local_matrix:
        sub[f"Q{qid}_local_matrix"] = [local_matrix] * len(sub)
        
    return sub.drop(columns=["answer"])

async def run_analysis(api_key: str, csv_path: str, out_base_path: str, log_callback, model_name: str = "gemini-1.5-pro-latest"):
    """執行完整分析流程的主函式"""
    # API金鑰現在為可選參數
    use_genai = bool(api_key and api_key.strip())
    
    if use_genai:
        if not configure_gemini(api_key, model_name):
            log_to_frontend("❌ API 金鑰設定失敗，請檢查金鑰是否正確。", log_callback)
            return
        log_to_frontend(f"🤖 使用模型: {model_name}", log_callback)
        log_to_frontend("🔗 將同時使用GenAI和非GenAI方法進行分析", log_callback)
    else:
        log_to_frontend("📊 只使用非GenAI方法進行分析", log_callback)

    df, qmap = load_exam(csv_path, log_callback)
    if df is None: return

    merged_df = df[["name"]].copy()
    
    for qid, col in qmap.items():
        res_df = await process_question(df, qid, col, log_callback, use_genai)
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

        # 儲存為 HTML（包含視覺化矩陣）
        html_content = generate_enhanced_html_report(merged_df)
        
        html_output = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>AI助教分析報告</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 20px; 
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 15px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 40px;
                    margin-bottom: 20px;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                }}
                .table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                }}
                .table th, .table td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                .table th {{ 
                    background-color: #3498db; 
                    color: white;
                    font-weight: bold;
                }}
                .table-striped tbody tr:nth-of-type(odd) {{ 
                    background-color: #f9f9f9; 
                }}
                .similarity-matrix {{
                    text-align: center;
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }}
                .matrix-image {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin: 10px;
                }}
                .matrix-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 20px;
                }}
                .matrix-item {{
                    flex: 1;
                    min-width: 300px;
                    max-width: 500px;
                }}
                .summary-stats {{
                    background: #e8f5e8;
                    border: 1px solid #27ae60;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎓 AI助教考卷分析報告</h1>
                {html_content}
            </div>
        </body>
        </html>
        """
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        log_to_frontend(f"🎉 HTML 報告已儲存至：\n{html_path}", log_callback)

    except Exception as e:
        log_to_frontend(f"❌ 儲存報告失敗: {e}", log_callback)

def generate_enhanced_html_report(merged_df):
    """
    生成包含視覺化矩陣的增強HTML報告
    """
    html_parts = []
    
    # 生成統計摘要
    total_students = len(merged_df)
    questions = [col for col in merged_df.columns if col.startswith('Q') and '_sim_flag' in col]
    html_parts.append(f"""
    <div class="summary-stats">
        <h2>📊 分析摘要</h2>
        <p><strong>總學生數:</strong> {total_students}</p>
        <p><strong>分析題目數:</strong> {len(questions)}</p>
        <p><strong>相似度檢測:</strong> {'已啟用' if questions else '未啟用'}</p>
    </div>
    """)
    
    # 為每個題目創建章節
    current_qid = None
    for col in merged_df.columns:
        if '_sim_flag' in col:
            qid = col.split('_')[0][1:]  # 提取Q後的數字
            if qid != current_qid:
                current_qid = qid
                html_parts.append(f"<h2>📝 題目 {qid} - 相似度分析</h2>")
                
                # 檢查是否有高相似度案例
                sim_flags = merged_df[f'Q{qid}_sim_flag'].tolist()
                high_sim_count = sum(1 for flag in sim_flags if flag == 2)
                medium_sim_count = sum(1 for flag in sim_flags if flag == 1)
                
                html_parts.append(f"""
                <div class="summary-stats">
                    <p><strong>高度相似:</strong> {high_sim_count} 位學生</p>
                    <p><strong>中等相似:</strong> {medium_sim_count} 位學生</p>
                    <p><strong>無明顯相似:</strong> {total_students - high_sim_count - medium_sim_count} 位學生</p>
                </div>
                """)
                
                # 添加相似度矩陣視覺化
                html_parts.append('<div class="similarity-matrix">')
                html_parts.append(f'<h3>相似度矩陣視覺化 - 題目 {qid}</h3>')
                html_parts.append('<div class="matrix-container">')
                
                # GenAI矩陣
                genai_col = f'Q{qid}_genai_matrix'
                local_col = f'Q{qid}_local_matrix'
                
                if genai_col in merged_df.columns and pd.notna(merged_df[genai_col].iloc[0]):
                    genai_matrix_b64 = merged_df[genai_col].iloc[0]
                    html_parts.append(f"""
                    <div class="matrix-item">
                        <h4>GenAI語義相似度</h4>
                        <img src="data:image/png;base64,{genai_matrix_b64}" class="matrix-image" alt="GenAI相似度矩陣">
                    </div>
                    """)
                
                if local_col in merged_df.columns and pd.notna(merged_df[local_col].iloc[0]):
                    local_matrix_b64 = merged_df[local_col].iloc[0]
                    html_parts.append(f"""
                    <div class="matrix-item">
                        <h4>非GenAI多算法相似度</h4>
                        <img src="data:image/png;base64,{local_matrix_b64}" class="matrix-image" alt="本地相似度矩陣">
                    </div>
                    """)
                
                html_parts.append('</div>')  # 關閉matrix-container
                html_parts.append('</div>')  # 關閉similarity-matrix
    
    # 生成完整數據表格
    html_parts.append('<h2>📋 完整分析數據</h2>')
    
    # 創建簡化的表格視圖（移除矩陣數據列）
    display_df = merged_df.copy()
    matrix_cols = [col for col in display_df.columns if '_matrix' in col]
    display_df = display_df.drop(columns=matrix_cols)
    
    table_html = display_df.to_html(index=False, escape=False, classes='table table-striped')
    html_parts.append(table_html)
    
    return '\n'.join(html_parts)

def create_similarity_matrix_visualization(texts: list[str], names: list[str] = None, question_id: str = ""):
    """
    創建相似度矩陣的視覺化圖表
    Returns: tuple (genai_matrix_b64, local_matrix_b64) - base64編碼的圖像
    """
    if len(texts) < 2:
        return None, None
    
    # 準備學生姓名標籤
    if names:
        labels = [name.split('(')[0].strip() if '(' in name else name[:10] for name in names]
    else:
        labels = [f"學生{i+1}" for i in range(len(texts))]
    
    # 截短標籤以適應顯示
    labels = [label[:8] + "..." if len(label) > 8 else label for label in labels]
    
    genai_matrix_b64 = None
    local_matrix_b64 = None
    
    # 1. GenAI相似度矩陣（如果有API）
    if gmodel:
        try:
            # 計算GenAI相似度矩陣
            processed_texts = [t.strip() if t and t.strip() else " " for t in texts]
            result = genai.embed_content(
                model="models/text-embedding-004", 
                content=processed_texts, 
                task_type="RETRIEVAL_DOCUMENT"
            )
            embs = result['embedding']
            genai_similarity_matrix = cosine_similarity(embs)
            
            # 創建GenAI矩陣視覺化
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(genai_similarity_matrix, dtype=bool))  # 只顯示下三角
            sns.heatmap(genai_similarity_matrix, 
                       mask=mask,
                       annot=True, 
                       fmt='.3f',
                       cmap='Reds',
                       xticklabels=labels,
                       yticklabels=labels,
                       cbar_kws={'label': '相似度'})
            
            plt.title(f'Q{question_id} - GenAI語義相似度矩陣', fontsize=14, pad=20)
            plt.xlabel('學生', fontsize=12)  
            plt.ylabel('學生', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # 轉為base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            genai_matrix_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
        except Exception as e:
            logging.error(f"GenAI相似度矩陣視覺化失敗: {e}")
    
    # 2. 非GenAI相似度矩陣
    try:
        from alternative_similarity_methods import calculate_text_similarity_enhanced
        
        # 計算本地相似度矩陣
        n = len(texts)
        local_similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j and texts[i].strip() and texts[j].strip():
                    from alternative_similarity_methods import calculate_advanced_similarity
                    similarity = calculate_advanced_similarity(texts[i], texts[j])
                    local_similarity_matrix[i][j] = similarity
                elif i == j:
                    local_similarity_matrix[i][j] = 1.0
        
        # 創建本地矩陣視覺化
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(local_similarity_matrix, dtype=bool))  # 只顯示下三角
        sns.heatmap(local_similarity_matrix,
                   mask=mask, 
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar_kws={'label': '相似度'})
        
        plt.title(f'Q{question_id} - 非GenAI多算法相似度矩陣', fontsize=14, pad=20)
        plt.xlabel('學生', fontsize=12)
        plt.ylabel('學生', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 轉為base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        local_matrix_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
    except Exception as e:
        logging.error(f"本地相似度矩陣視覺化失敗: {e}")
    
    return genai_matrix_b64, local_matrix_b64