// 全域變數來儲存選擇的檔案資訊
let selectedFile = null;

// 當 pywebview API 準備就緒時，初始化所有事件監聽器
window.addEventListener('pywebviewready', () => {
    const apiKeyInput = document.getElementById('api-key');
    const modelSelect = document.getElementById('model-select');
    const dropZone = document.getElementById('drop-zone');
    const dropZoneText = document.getElementById('drop-zone-text');
    const startButton = document.getElementById('start-button');
    const startButtonText = startButton.querySelector('span');

    // 檢查是否所有必要輸入都已備妥，並更新按鈕狀態
    const checkButtonState = () => {
        const isReady = apiKeyInput.value.trim();
        startButton.disabled = !isReady;
    };

    apiKeyInput.addEventListener('input', checkButtonState);

    // --- 檔案選擇按鈕事件 ---
    const selectFileButton = document.createElement('button');
    selectFileButton.textContent = '選擇檔案';
    selectFileButton.className = 'file-select-btn';
    selectFileButton.addEventListener('click', async () => {
        try {
            const filePath = await window.pywebview.api.select_file();
            if (filePath) {
                const fileName = filePath.split(/[/\\]/).pop();
                selectedFile = { path: filePath, name: fileName };
                dropZoneText.textContent = `已選擇檔案：${fileName}`;
                dropZone.style.borderColor = 'var(--success-color)';
            }
        } catch (error) {
            console.error('檔案選擇錯誤:', error);
            appendToLog('檔案選擇失敗，請重試。');
        }
        checkButtonState();
    });

    // 將選擇檔案按鈕添加到拖曳區域
    dropZone.appendChild(selectFileButton);

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
    startButton.addEventListener('click', async () => {
        const apiKey = apiKeyInput.value.trim();
        const selectedModel = modelSelect.value;
        
        if (!apiKey) {
            alert('請輸入 API 金鑰。');
            return;
        }

        // 禁用按鈕並更新顯示文字
        startButton.disabled = true;
        startButtonText.textContent = '分析中，請稍候...';
        clearLog();
        
        const filePath = selectedFile ? selectedFile.path : '';
        
        try {
            // 呼叫 Python 後端函式 (新版 API)
            const result = await window.pywebview.api.start_analysis({ 
                apiKey: apiKey, 
                filePath: filePath,
                modelName: selectedModel
            });
            
            if (result.status === 'error') {
                appendToLog(`❌ ${result.message}`);
                startButtonText.textContent = '開始分析';
                startButton.disabled = false;
            } else {
                appendToLog('🚀 分析已開始...');
            }
        } catch (error) {
            console.error('分析啟動錯誤:', error);
            appendToLog('❌ 分析啟動失敗，請檢查API金鑰和檔案。');
            startButtonText.textContent = '開始分析';
            startButton.disabled = false;
        }
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
