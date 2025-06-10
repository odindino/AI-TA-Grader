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

    def start_analysis(self, api_key, file_path):
        """
        由前端呼叫，在一個獨立的執行緒中開始分析，以避免 GUI 凍結。
        """
        thread = threading.Thread(target=self._run_analysis_in_thread, args=(api_key, file_path))
        thread.start()

    def _run_analysis_in_thread(self, api_key, file_path):
        """
        執行緒的目標函式。它會設定一個新的 asyncio 事件循環，並執行分析任務。
        """
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
            loop.run_until_complete(run_analysis(api_key, file_path, unique_output_base_name, self._log_to_frontend))
        finally:
            loop.close()
            # 通知前端分析已完成
            if self.window:
                self.window.evaluate_js('analysis_complete()')
    
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
