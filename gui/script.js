// 全域變數來儲存選擇的檔案資訊
let selectedFile = null;

// 當 pywebview API 準備就緒時，初始化所有事件監聽器
window.addEventListener('pywebviewready', () => {
    const apiKeyInput = document.getElementById('api-key');
    const dropZone = document.getElementById('drop-zone');
    const dropZoneText = document.getElementById('drop-zone-text');
    const startButton = document.getElementById('start-button');
    const startButtonText = startButton.querySelector('span');

    // 檢查是否所有必要輸入都已備妥，並更新按鈕狀態
    const checkButtonState = () => {
        const isReady = apiKeyInput.value.trim() && selectedFile;
        startButton.disabled = !isReady;
    };

    apiKeyInput.addEventListener('input', checkButtonState);

    // --- 拖曳區域事件處理 ---
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            // 驗證檔案類型
            if (file.name.toLowerCase().endsWith('.csv')) {
                selectedFile = { path: file.path, name: file.name };
                dropZoneText.textContent = `已選擇檔案：${file.name}`;
                dropZone.style.borderColor = 'var(--success-color)';
            } else {
                selectedFile = null;
                dropZoneText.textContent = '錯誤：請選擇 CSV 檔案';
                dropZone.style.borderColor = 'var(--error-color)';
            }
        }
        checkButtonState();
    });

    // --- 開始按鈕點擊事件 ---
    startButton.addEventListener('click', () => {
        const apiKey = apiKeyInput.value.trim();
        if (!apiKey || !selectedFile) {
            alert('請確認已輸入 API 金鑰並選擇了 CSV 檔案。');
            return;
        }

        // 禁用按鈕並更新顯示文字
        startButton.disabled = true;
        startButtonText.textContent = '分析中，請稍候...';
        clearLog();
        appendToLog('🚀 初始化分析流程...');

        // 呼叫 Python 後端函式
        window.pywebview.api.start_analysis(apiKey, selectedFile.path);
    });
});

// --- 由 Python 呼叫的前端函式 ---

/**
 * 將訊息附加到日誌輸出區域
 * @param {string} message - 要顯示的訊息
 */
function appendToLog(message) {
    const logOutput = document.getElementById('log-output');
    logOutput.textContent += message + '\n';
    logOutput.scrollTop = logOutput.scrollHeight; // 自動滾動到底部
}

/**
 * 清空日誌區域
 */
function clearLog() {
    document.getElementById('log-output').textContent = '';
}

/**
 * 當分析完成後，由 Python 呼叫此函式以重設按鈕狀態
 */
function analysis_complete() {
    const startButton = document.getElementById('start-button');
    const startButtonText = startButton.querySelector('span');
    startButtonText.textContent = '分析完成，可再次執行';
    // 短暫延遲後再重新啟用按鈕，給予使用者回饋感
    setTimeout(() => {
        startButton.disabled = false;
        startButtonText.textContent = '開始分析';
    }, 2000);
}
