# 開發紀錄 (Development Log)

## 2025-06-10 系統優化與功能重構

### 🎯 開發目標
- 系統全面檢查和功能驗證
- 修復報告生成中的缺失資訊
- 優化相似度檢測策略
- 新增日誌記錄功能

### 📋 問題識別與解決

#### 1. 初始問題檢測
**問題描述**：HTML報告缺少重要資訊
- 缺少AI工具作答嫌疑度評估
- GenAI和local抄襲度分析不完整
- 所有分數顯示為0分
- 沒有AI輸出記錄

**解決方案**：
- 修復 `backend/gemini_client.py` 中 `grade_block` 參數缺失問題
- 改善 `backend/analyzer.py` 本地評分算法，提供合理的分數範圍
- 新增AI風險評估功能到HTML報告中

#### 2. 系統架構優化

**matplotlib 線程問題修復**：
```python
# backend/visualization.py
matplotlib.use('Agg')  # 使用非GUI後端，避免線程問題
```

**Gemini API整合改善**：
```python
# 修復前
batch_scores = await self.gemini_client.grade_responses_batch(batch_texts, rubric, qid)

# 修復後  
batch_scores, batch_ai_risks = await self.gemini_client.grade_responses_batch(batch_texts, rubric, qid)
```

#### 3. 相似度檢測策略重構

**原始策略**：AI + 本地雙重檢測，輸出 0-2 標記
**新策略**：純本地檢測，輸出 0-100 分數

**主要修改**：
- `backend/similarity_detector.py`：移除AI相似度檢測，強制使用本地算法
- `backend/analyzer.py`：將 `similarity_flags` 改為 `similarity_scores`
- `backend/visualization.py`：移除GenAI相似度矩陣，保留本地算法矩陣

```python
# 舊版本
def _calculate_flags_from_matrix(self, matrix: np.ndarray) -> List[int]:
    return [2 if score > 0.7 else 1 if score > 0.3 else 0 for score in max_scores]

# 新版本  
def _calculate_scores_from_matrix(self, matrix: np.ndarray) -> List[int]:
    return [int(score * 100) for score in max_scores]
```

#### 4. 日誌系統實作

**新增功能**：完整的控制台輸出記錄到檔案
- 實作 `CombinedLogHandler` 類別
- 同時輸出到前端和儲存到檔案
- 檔案命名格式：`AI_TA_Grader_Log_YYYYMMDD_HHMMSS.log`

```python
# app.py 中的日誌處理器
class CombinedLogHandler(logging.Handler):
    def __init__(self, callback, log_file_path):
        super().__init__()
        self.callback = callback
        self.log_file_path = log_file_path
        self.log_messages = []
    
    def emit(self, record):
        msg = self.format(record)
        # 發送到前端
        if self.callback:
            self.callback(msg)
        # 記錄到內存中
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {msg}"
        self.log_messages.append(log_entry)
```

### 🔧 技術修改摘要

#### 檔案修改清單
1. **app.py**
   - 新增 `CombinedLogHandler` 類別
   - 整合日誌記錄功能
   - 移除xlsx報告輸出

2. **backend/analyzer.py**
   - 改善本地評分算法 (`_estimate_score_local`)
   - 新增關鍵詞加分機制
   - 修改相似度處理邏輯

3. **backend/gemini_client.py**
   - 修復 `grade_block` 參數問題
   - 新增 AI 風險評估功能
   - 改善錯誤處理機制

4. **backend/similarity_detector.py**
   - 移除 AI 相似度檢測
   - 強制使用本地算法（`use_genai=False`）
   - 輸出 0-100 分數而非 0-2 標記

5. **backend/visualization.py**
   - 移除 GenAI 相似度矩陣生成
   - 新增詳細相似度分析表格
   - 改善 HTML 報告格式

### 📊 功能驗證結果

#### 測試檔案
- **輸入**：`Final_Report_Updated_Test.csv` (3位學生，半導體工程考試)
- **問題內容**：
  - Q1: Czochralski vs Floating Zone 製程比較
  - Q2: CVD 機制分析和 MBE 原理

#### 相似度檢測結果
- **Q1 相似度分數**：[82, 11, 86] - 檢測到高相似度
- **Q2 相似度分數**：[8, 9, 9] - 低相似度，正常範圍

#### API 測試結果
- **Gemini 2.5 Pro Exp**：❌ 無免費配額
- **Gemini 1.5 Pro**：❌ 配額已用盡  
- **Gemini 1.5 Flash**：✅ 成功運行

### 🎨 報告改善

#### HTML 報告新增功能
1. **AI 使用嫌疑度表格**：色彩編碼顯示風險等級
2. **詳細相似度分析表**：0-100分數制，四級顏色分類
3. **視覺化矩陣**：僅保留本地算法矩陣
4. **完整數據表**：包含所有分數和風險評估

#### 色彩編碼標準
- **AI 風險**：高(≥70)紅色、中(40-69)橘色、低(<40)綠色
- **相似度**：高(≥85)紅色、中(70-84)橘色、低(50-69)黃色、無(<50)綠色

### 🚀 系統運行狀態

#### 當前配置
- **API 模型**：Gemini 1.5 Flash (推薦)
- **相似度檢測**：純本地算法
- **評分機制**：AI評分 + 本地後備
- **報告格式**：CSV + HTML
- **日誌記錄**：完整控制台輸出

#### 性能表現
- ✅ 本地相似度檢測：正常運作
- ✅ AI 評分系統：使用 1.5 Flash 成功
- ✅ AI 風險評估：正常產生
- ✅ 報告生成：完整且準確
- ✅ 日誌記錄：成功儲存

### 🔧 2025-06-11 深夜更新：多題目與大班級支援

#### 問題發現
使用真實學生數據測試時發現：
1. **問題欄位檢測限制**：系統只處理前2題，無法處理12題完整考試
2. **視覺化文字重疊**：20+學生時矩陣圖標籤嚴重重疊
3. **API安全過濾**：Gemini API的safety filter導致所有請求失敗
4. **缺少數據表格**：只有視覺化圖片，缺少原始數據表格

#### 解決方案實施

**1. 自動問題檢測機制**
```python
# backend/analyzer.py - 自動檢測所有問題欄位
def _find_question_columns(self, df: pd.DataFrame, target_questions: List[int] = None):
    # 自動檢測包含問題ID的欄位（352xxx, 362xxx等Canvas格式）
    potential_question_cols = []
    for col in df.columns:
        if (col_str.startswith('352') or col_str.startswith('362') or 
            col_str.startswith('Q') or any(char.isdigit() for char in col_str[:10])):
            potential_question_cols.append(col)
```

**2. 視覺化自適應調整**
- 根據學生數量動態調整圖片尺寸
- 20+學生：極短標籤(3字元)、90度旋轉、大圖尺寸
- 15-20學生：中等標籤(5字元)、60度旋轉
- <15學生：正常顯示

**3. API錯誤處理改善**
```python
# backend/gemini_client.py - 檢查回應狀態
if not response.candidates or not response.candidates[0].content.parts:
    finish_reason = response.candidates[0].finish_reason
    self.logger.warning(f"API回應被過濾 (finish_reason: {finish_reason})")
    # 使用本地評分作為後備
    local_score = self._estimate_local_score(text)
    return local_score, 0
```

**4. 相似度矩陣數據表格**
- 新增 `_generate_similarity_matrix_table()` 函數
- 顯示完整的相似度數值矩陣
- 色彩編碼便於快速識別

#### 測試結果
- ✅ 成功處理12題完整考試
- ✅ 25+學生的視覺化清晰可讀
- ✅ API失敗時優雅降級
- ✅ 同時提供圖表和數據表格

### 💡 建議與注意事項

#### API 使用建議
1. **首選模型**：`gemini-1.5-flash-latest` (較寬鬆的配額限制)
2. **備選模型**：`gemini-1.5-pro-latest` (更高品質但配額較嚴)
3. **避免使用**：`gemini-2.5-pro-exp` (無免費配額)

#### 系統特性
- **混合模式**：可在有/無 API 金鑰情況下運行
- **graceful degradation**：API 失敗時自動切換本地評分
- **工業級相似度檢測**：6種算法組合，無需依賴AI
- **完整記錄**：所有操作都有詳細日誌

### 📈 未來開發方向
1. **批次處理優化**：提升大量資料處理效率
2. **評分標準客製化**：允許使用者自定義評分rubrics
3. **多語言支援**：擴展到其他語言的文本分析
4. **進階相似度算法**：加入更多學術抄襲檢測方法

---

### 📊 系統特色與應用場景

#### 主要應用
- **NTUCOOL系統整合**：專為批改NTUCOOL(臺大酷)平台的學生考試回答設計
- **Canvas LMS相容**：支援Canvas問題ID格式(352xxx, 362xxx等)
- **大班級支援**：可處理50+學生的考試數據
- **多題型處理**：自動檢測並處理所有問題欄位

#### 技術亮點
- **混合評分模式**：結合AI評分與本地算法
- **工業級相似度檢測**：6種算法組合，準確率高
- **自適應視覺化**：根據班級規模自動調整
- **完整審計追蹤**：詳細的日誌記錄系統

---

**開發者**：odindino  
**初版日期**：2025-06-10  
**最後更新**：2025-06-11  
**版本**：Enhanced v2.1 (Multi-Question Support)  
**狀態**：Production Ready ✅