# AI-TA-Grader

AI-TA-Grader 是一個專為教師與助教設計的桌面應用程式，利用 Google Gemini 的強大語言模型，協助分析學生申論題答案、評估 AI 書寫風格，並根據預設評分標準（Rubric）進行初步評分，旨在提升學術工作的效率與公正性。

## 功能特色

* **自動化申論題分析**：利用 AI 技術深入理解學生答案的語義和結構。
* **AI 書寫風格檢測**：評估文本是否可能由 AI生成，協助維護學術誠信。
* **基於 Rubric 的評分**：根據使用者自訂的評分標準進行初步評分。
* **易用的桌面應用程式**：提供直觀的圖形使用者介面，方便操作。
* **結果匯出**：可將分析報告匯出為 Excel 檔案，方便後續處理。

## 技術棧

* **後端**：Python, Google Gemini API
* **GUI**：pywebview (HTML, CSS, JavaScript)
* **資料處理**：pandas

## 安裝指南

1. **複製專案庫**：

    ```bash
    git clone https://github.com/your-username/AI-TA-Grader.git
    cd AI-TA-Grader
    ```

2. **建立虛擬環境並啟用** (建議)：

    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate  # Windows
    ```

3. **安裝依賴套件**：

    ```bash
    pip install -r requirements.txt
    ```

    或者，如果您使用 Conda：

    ```bash
    conda env create -f environment.yml
    conda activate ai_ta_grader_env
    ```

4. **設定 API 金鑰**：
    您需要在應用程式中輸入您的 Google Gemini API 金鑰才能使用分析功能。

## 使用方法

1. 執行應用程式：

    ```bash
    python app.py
    ```

2. 在應用程式介面中，輸入您的 Google Gemini API 金鑰。
3. 選擇包含學生答案的檔案（支援格式待確認，目前推測為純文字或 CSV/Excel）。
4. 點擊「開始分析」按鈕。
5. 分析完成後，報告將會儲存（預設為原檔案目錄下的 `*_report.xlsx`），並且前端介面會顯示完成訊息。

## 未來展望

* 支援更多檔案格式輸入。
* 提供更詳細的客製化評分標準設定。
* 使用者帳戶系統與歷史紀錄。

## 貢獻

歡迎各種形式的貢獻！如果您有任何建議或發現錯誤，請隨時提出 Issue 或 Pull Request。
