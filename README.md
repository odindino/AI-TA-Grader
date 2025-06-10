# AI-TA-Grader 🎓✨ (v2.0 重構版)

AI-TA-Grader 是一個專為教師與助教設計的桌面應用程式，利用 Google Gemini 的強大語言模型，協助分析學生申論題答案、評估 AI 書寫風格，並根據預設評分標準（Rubric）進行初步評分，旨在提升學術工作的效率與公正性。

## 🆕 v2.0 重構版新特性

- 🏗️ **模組化架構**：全新的 backend 模組設計，易於維護和擴展
- 🔑 **可選 API 金鑰**：支援有/無 API 金鑰兩種運行模式
- 🧠 **工業級相似度檢測**：升級為 6 種先進算法的加權組合
- 📈 **美觀視覺化**：專業的相似度矩陣熱力圖
- 📋 **增強報告**：包含統計摘要和嵌入式視覺化的 HTML 報告
- 🧪 **完整測試**：15 個測試檔案確保系統穩定性

## 核心功能

🤖 **AI 書寫風格偵測**：對每份答案產生一個「AI 風險分數」（0-100），分數越高代表書寫風格越像 AI 生成。（需要API金鑰）

⚖️ **自動化評分輔助**：根據使用者自訂的評分標準（Rubrics），對指定問題進行評分。（需要API金鑰）

🕵️ **答案相似度比對**：利用多種先進算法檢測學生答案之間的相似度，包括：
   - **GenAI語義分析**：使用Google text-embedding-004模型進行深度語義比對（需要API金鑰）
   - **非GenAI多算法檢測**：結合6種工業級算法：LCS、編輯距離、語義塊、增強TF-IDF、字符級相似度、詞彙重疊（無需API金鑰）

📊 **視覺化相似度矩陣**：為每個題目生成直觀的相似度熱力圖，快速識別潛在的抄襲關係。

🔧 **靈活運行模式**：
   - **完整模式**：提供API金鑰時，同時執行GenAI和非GenAI分析
   - **離線模式**：無API金鑰時，僅執行非GenAI相似度檢測（同樣具有工業級檢測能力）

🖥️ **直觀的圖形化介面**：支援拖曳上傳 CSV 檔案，操作簡單，無需編寫任何程式碼。

🔄 **自動化報告生成**：分析完成後，自動生成包含所有分數與旗標的 Excel/HTML/CSV 報告，並確保檔案名稱不重複。

🔒 **本地端執行**：所有操作均在您的電腦上完成，學生資料與 API 金鑰不會上傳至任何第三方伺服器。

## 技術架構

* **後端**: Python 模組化架構
  - `backend/analyzer.py` - 主要分析引擎
  - `backend/gemini_client.py` - Gemini API 客戶端
  - `backend/similarity_detector.py` - 相似度檢測器
  - `backend/visualization.py` - 視覺化引擎
  - `backend/data_processor.py` - 數據處理器
* **前端**: HTML, CSS, JavaScript  
* **GUI 框架**: pywebview
* **AI 模型**: Google Gemini API (gemini-1.5-pro & text-embedding-004)
* **核心函式庫**: pandas, numpy, scikit-learn, matplotlib, seaborn
* **相似度檢測**: 6種先進算法加權組合

## 安裝與設定

### 1. 前置需求

* Python 3.12 或更高版本。
* Anaconda 或 Miniconda (若您選擇 Anaconda 環境)。

### 2. 下載專案

從 GitHub 下載或 clone 此專案：

```bash
git clone https://github.com/odindino/AI-TA-Grader.git
cd AI-TA-Grader
```

### 3. 設定環境與安裝依賴套件

我們提供兩種主流的環境設定方式： Anaconda 或 pip + venv。請擇一即可。

#### 選項 A：使用 Anaconda (推薦)

如果您習慣使用 Anaconda，我們已提供 `environment.yml` 檔案來快速建立環境。

```bash
# 使用 environment.yml 建立 Conda 環境
conda env create -f environment.yml

# 啟用新的 Conda 環境
conda activate ai-ta-grader
```

#### 選項 B：使用 pip 與 venv

若您不使用 Anaconda，也可以使用 Python 內建的 `venv` 來建立虛擬環境。

```bash
# 建立虛擬環境
python -m venv venv

# 啟用虛擬環境
# on Windows:
# venv\Scripts\activate
# on macOS/Linux:
source venv/bin/activate

# 安裝所有必要的函式庫
pip install -r requirements.txt
```

### 4. 設定 API 金鑰

本工具需要您自己的 Google Gemini API 金鑰。您可以從 Google AI Studio 獲取。
無需在程式碼中填寫金鑰，您只需在啟動應用程式後的介面中輸入即可。

## 使用教學

1. **啟用您的環境**:

    ```bash
    # 如果使用 Anaconda
    conda activate ai-ta-grader

    # 如果使用 venv
    source venv/bin/activate
    ```

2. **啟動應用程式**:

    ```bash
    python app.py
    ```

3. **輸入 API 金鑰**: 在應用程式視窗中，將您的 Gemini API Key 貼到指定欄位。
4. **上傳檔案**: 將包含學生答案的 `.csv` 檔案直接拖曳到視窗中的虛線框區域。
5. **開始分析**: 確認金鑰與檔案都已就緒後，點擊「開始分析」按鈕。
6. **等待結果**: 處理進度會即時顯示在下方的「處理日誌」區域。分析完成後，將自動生成包含視覺化相似度矩陣的多格式報告 (`.xlsx`, `.csv`, `.yaml`, `.html`)。

### 💡 新功能亮點

**🔧 彈性運行模式**:
- **完整模式**: 提供API金鑰時，執行GenAI分析 + 非GenAI檢測
- **離線模式**: 無API金鑰時，僅執行工業級非GenAI相似度檢測

**📊 視覺化相似度分析**: 
- 每個題目都會生成直觀的相似度熱力圖
- 支援GenAI語義分析和非GenAI多算法檢測的雙重矩陣
- 快速識別學生間的相似關係

**🎯 工業級檢測算法**:
- 結合6種先進相似度算法（TF-IDF、編輯距離、LCS、N-gram等）
- 參考Turnitin等商業系統的核心技術
- 即使無API金鑰也具備專業級檢測能力

## 專案結構

```text
AI-TA-Grader/
├── app.py              # 主應用程式，啟動 GUI
├── analyzer.py         # 後端分析邏輯
├── requirements.txt    # pip 依賴套件列表
├── environment.yml     # Conda 環境設定檔
└── gui/
    ├── index.html      # 前端介面骨架
    ├── style.css       # 介面樣式
    └── script.js       # 前端互動邏輯
```

## 授權條款

本專案採用 MIT License 授權。

## 如何貢獻

歡迎任何形式的貢獻！如果您有任何建議、發現 bug，或是想新增功能，請隨時提出 Issue 或發送 Pull Request。

Made with ❤️ for a more efficient and fair academic world.
