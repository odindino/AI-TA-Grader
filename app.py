import webview
import threading
import asyncio
import os
import re
from analyzer import run_analysis # 從 analyzer.py 匯入後端邏輯

class Api:
    """
    這個類別的方法將會暴露給前端 JavaScript。
    """
    def __init__(self):
        self.window = None

    def _get_unique_filepath(self, base_path):
        """
        生成唯一的檔案路徑，如果檔案已存在，則在檔名後加上 _1, _2...
        """
        directory, filename = os.path.split(base_path)
        name, ext = os.path.splitext(filename)
        
        # 移除檔名中可能已存在的舊數字後綴 (例如 report_1)
        name = re.sub(r'_\d+$', '', name)
        
        # 如果原始路徑不存在，直接使用
        if not os.path.exists(base_path):
            return base_path
            
        # 否則，開始尋找可用的檔名
        counter = 1
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        while os.path.exists(new_path):
            counter += 1
            new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        return new_path

    def select_file(self):
        """
        開啟檔案選擇對話框
        """
        file_types = ('CSV Files (*.csv)', 'All files (*.*)')
        result = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=file_types
        )
        if result and len(result) > 0:
            return result[0]  # 返回選擇的檔案路徑
        return None

    def get_available_models(self):
        """取得可用的 Gemini 模型列表"""
        try:
            # 推薦的模型列表 (基於檢查結果)
            recommended_models = [
                "gemini-1.5-pro-latest",
                "gemini-1.5-pro-002", 
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash-002",
                "gemini-2.0-flash",
                "gemini-2.0-flash-001",
                "gemini-2.0-pro-exp",
                "gemini-2.5-pro-preview-06-05",
                "gemini-exp-1206"
            ]
            return {"status": "success", "models": recommended_models}
        except Exception as e:
            return {"status": "error", "message": f"無法獲取模型列表: {str(e)}"}

    def start_analysis(self, params):
        """
        由前端呼叫，在一個獨立的執行緒中開始分析，以避免 GUI 凍結。
        """
        api_key = params.get('apiKey', '').strip()
        file_path = params.get('filePath', '').strip()
        model_name = params.get('modelName', 'gemini-1.5-pro-latest').strip()
        
        # 輸入驗證
        if not api_key:
            self._log_to_frontend("❌ 請先輸入 API 金鑰。")
            return {"status": "error", "message": "API 金鑰未提供"}
        
        # 如果沒有提供檔案路徑，使用預設測試檔案
        if not file_path:
            default_file = os.path.join(os.path.dirname(__file__), 'testfile', 'Final Exam Quiz Student Analysis Report_Public.csv')
            if os.path.exists(default_file):
                file_path = default_file
                self._log_to_frontend(f"📁 使用預設測試檔案: {os.path.basename(default_file)}")
            else:
                self._log_to_frontend("❌ 請先選擇或拖曳 CSV 檔案。")
                return {"status": "error", "message": "檔案路徑未提供"}
        
        # 檢查檔案是否存在
        if not os.path.exists(file_path):
            self._log_to_frontend(f"❌ 找不到檔案: {file_path}")
            return {"status": "error", "message": "檔案不存在"}
        
        # 檢查檔案是否為 CSV
        if not file_path.lower().endswith('.csv'):
            self._log_to_frontend("❌ 請選擇 CSV 格式的檔案。")
            return {"status": "error", "message": "檔案格式不正確"}
            
        self._log_to_frontend(f"🚀 開始分析檔案: {os.path.basename(file_path)}")
        self._log_to_frontend(f"🤖 使用模型: {model_name}")
        
        thread = threading.Thread(target=self._run_analysis_in_thread, args=(api_key, file_path, model_name))
        thread.start()
        
        return {"status": "success", "message": "分析已開始"}

    def _run_analysis_in_thread(self, api_key, file_path, model_name):
        """
        執行緒的目標函式。它會設定一個新的 asyncio 事件循環，並執行分析任務。
        """
        try:
            # 建立唯一的輸出檔案路徑
            output_dir = os.path.dirname(file_path)
            base_input_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # 為主要的 .xlsx 報告產生唯一的檔案路徑
            xlsx_output_path = os.path.join(output_dir, f"{base_input_name}_report.xlsx")
            unique_xlsx_output_path = self._get_unique_filepath(xlsx_output_path)
            
            # 從唯一的 .xlsx 路徑中提取不含副檔名的基本名稱 (例如 /path/to/file_report_1)
            # 這個基本名稱將用於所有格式的報告檔案
            unique_output_base_name = os.path.splitext(unique_xlsx_output_path)[0]

            # 每個執行緒都需要自己的 asyncio 事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # 執行非同步的分析主函式
                loop.run_until_complete(run_analysis(api_key, file_path, unique_output_base_name, self._log_to_frontend, model_name))
                self._log_to_frontend("✅ 分析完成！")
            finally:
                loop.close()
                # 通知前端分析已完成
                if self.window:
                    self.window.evaluate_js('analysis_complete()')
        except Exception as e:
            self._log_to_frontend(f"❌ 分析過程中發生錯誤: {str(e)}")
    def _log_to_frontend(self, message: str):
        """
        一個簡單的回呼函式，用於從後端傳遞字串訊息到前端。
        """
        if self.window:
            # 清理訊息中的特殊字元，避免 JavaScript 錯誤
            escaped_message = message.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
            self.window.evaluate_js(f"appendToLog('{escaped_message}')")

def main():
    api = Api()
    window = webview.create_window(
        'AI 助教考卷分析工具',
        'gui/index.html',  # 相對路徑指向 HTML 檔案
        js_api=api,
        width=750,
        height=780,
        resizable=True # 允許調整視窗大小
    )
    # 將 window 物件設定給 API 實例，以便在後端呼叫 JS
    api.window = window
    webview.start()

if __name__ == '__main__':
    main()
