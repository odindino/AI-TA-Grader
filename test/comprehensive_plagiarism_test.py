#!/usr/bin/env python3
"""
完整的抄襲檢測測試 - 包含視覺化和HTML報告生成
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime

# 添加項目根目錄到Python路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_similarity_heatmap(similarity_matrix, student_names, question_title):
    """創建相似度熱力圖"""
    plt.figure(figsize=(10, 8))
    
    # 創建遮罩，只顯示下三角
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
    
    # 創建熱力圖
    sns.heatmap(similarity_matrix,
                mask=mask,
                annot=True,
                fmt='.3f',
                cmap='Reds',
                xticklabels=student_names,
                yticklabels=student_names,
                cbar_kws={'label': '相似度'},
                square=True)
    
    plt.title(f'相似度矩陣 - {question_title[:50]}...', fontsize=14, pad=20)
    plt.xlabel('學生', fontsize=12)
    plt.ylabel('學生', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 轉為base64字符串
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def generate_html_report(results, output_path):
    """生成HTML抄襲檢測報告"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI-TA-Grader 抄襲檢測報告</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 3px solid #e74c3c;
            }}
            .header h1 {{
                color: #2c3e50;
                margin: 0;
                font-size: 2.5em;
            }}
            .summary {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .summary h2 {{
                color: #e74c3c;
                margin-top: 0;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-card {{
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #e74c3c;
            }}
            .stat-label {{
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            .question-section {{
                margin-bottom: 40px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
            }}
            .question-title {{
                color: #2c3e50;
                font-size: 1.3em;
                margin-bottom: 20px;
                padding: 10px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }}
            .similarity-matrix {{
                text-align: center;
                margin: 20px 0;
            }}
            .similarity-matrix img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .high-similarity-pairs {{
                background-color: #fff5f5;
                border: 1px solid #fbb3b3;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            }}
            .high-similarity-pairs h4 {{
                color: #e74c3c;
                margin-top: 0;
            }}
            .pair-info {{
                background-color: white;
                margin: 10px 0;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #e74c3c;
            }}
            .similarity-score {{
                font-weight: bold;
                color: #e74c3c;
                font-size: 1.2em;
            }}
            .answer-preview {{
                background-color: #f8f9fa;
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
                font-family: monospace;
                font-size: 0.9em;
                border-left: 3px solid #6c757d;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
            }}
            .alert {{
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
                border: 1px solid #f5c6cb;
                background-color: #f8d7da;
                color: #721c24;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 AI-TA-Grader 抄襲檢測報告</h1>
                <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>檢測模式: 離線模式（6種工業級算法）</p>
            </div>
            
            <div class="summary">
                <h2>📊 檢測摘要</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{results['total_students']}</div>
                        <div class="stat-label">學生總數</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{results['total_questions']}</div>
                        <div class="stat-label">問題總數</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{results['total_high_similarity_pairs']}</div>
                        <div class="stat-label">高相似度對總數</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{results['max_similarity']:.1%}</div>
                        <div class="stat-label">最高相似度</div>
                    </div>
                </div>
            </div>
    """
    
    if results['total_high_similarity_pairs'] > 0:
        html_content += '''
            <div class="alert">
                <strong>⚠️ 警告:</strong> 檢測到可疑的高相似度答案，請仔細審查下方詳細結果。
            </div>
        '''
    
    # 添加每個問題的詳細結果
    for i, question_result in enumerate(results['questions']):
        html_content += f'''
            <div class="question-section">
                <div class="question-title">
                    問題 {i+1}: {question_result['title'][:100]}...
                </div>
                
                <div class="similarity-matrix">
                    <h4>相似度矩陣</h4>
                    <img src="data:image/png;base64,{question_result['heatmap']}" alt="相似度矩陣">
                </div>
        '''
        
        if question_result['high_similarity_pairs']:
            html_content += f'''
                <div class="high-similarity-pairs">
                    <h4>🚨 高相似度對 (>{70}%)</h4>
            '''
            
            for pair in question_result['high_similarity_pairs']:
                html_content += f'''
                    <div class="pair-info">
                        <div class="similarity-score">
                            {pair['student1']} ↔ {pair['student2']}: {pair['similarity']:.1%}
                        </div>
                        <div class="answer-preview">
                            <strong>{pair['student1']}:</strong> {pair['answer1'][:150]}...
                        </div>
                        <div class="answer-preview">
                            <strong>{pair['student2']}:</strong> {pair['answer2'][:150]}...
                        </div>
                    </div>
                '''
            
            html_content += '''
                </div>
            '''
        else:
            html_content += '''
                <div style="color: #27ae60; padding: 15px; background-color: #d5f4e6; border-radius: 5px;">
                    ✅ 此問題未發現高相似度答案
                </div>
            '''
        
        html_content += '''
            </div>
        '''
    
    html_content += f'''
            <div class="footer">
                <p>AI-TA-Grader v2.0 - 自動化教學助理評分系統</p>
                <p>使用 6 種工業級相似度檢測算法進行分析</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML報告已生成: {output_path}")

def main():
    print("🔍 完整抄襲檢測測試（含視覺化和報告）")
    print("=" * 60)
    
    try:
        # 匯入模組
        from backend.data_processor import DataProcessor
        from backend.similarity_detector import SimilarityDetector
        print("✅ 模組匯入成功")
        
        # 初始化組件
        similarity_detector = SimilarityDetector()
        print("✅ 組件初始化成功")
        
        # 載入測試資料
        test_file = project_root / "testfile" / "Final Exam Quiz Student Analysis Report_Public_plag.csv"
        df = pd.read_csv(test_file)
        print(f"✅ 載入測試檔案: {len(df)} 筆資料")
        
        # 找出問題欄位
        exclude_cols = ['name', 'id', 'sis_id', 'section', 'section_id', 'section_sis_id', 
                       'submitted', 'attempt', '10.0', '10.0.1', 'n correct', 'n incorrect', 'score']
        question_cols = [col for col in df.columns 
                        if '352902' in col or '352903' in col or 
                        (col not in exclude_cols and len(col) > 50)]
        
        print(f"📝 找到 {len(question_cols)} 個問題欄位")
        
        # 準備學生名稱
        student_names = [f"學生 {i}" for i in range(len(df))]
        
        # 準備結果數據
        results = {
            'total_students': len(df),
            'total_questions': len(question_cols),
            'total_high_similarity_pairs': 0,
            'max_similarity': 0.0,
            'questions': []
        }
        
        # 分析每個問題
        for i, question_col in enumerate(question_cols):
            print(f"\n🔍 分析問題 {i+1}: {question_col[:50]}...")
            
            # 取得回答
            answers = df[question_col].fillna("").astype(str).tolist()
            
            # 計算相似度
            result = similarity_detector.calculate_local_similarity(answers)
            
            if result and result['info']['status'] == 'success':
                similarity_matrix = result['matrix']
                
                # 創建熱力圖
                heatmap_b64 = create_similarity_heatmap(
                    similarity_matrix, student_names, question_col
                )
                
                # 找出高相似度對
                high_similarity_pairs = []
                for row_i in range(len(similarity_matrix)):
                    for col_j in range(row_i + 1, len(similarity_matrix[0])):
                        similarity = similarity_matrix[row_i][col_j]
                        
                        if similarity > 0.7:
                            high_similarity_pairs.append({
                                'student1': student_names[row_i],
                                'student2': student_names[col_j],
                                'similarity': similarity,
                                'answer1': answers[row_i],
                                'answer2': answers[col_j]
                            })
                            results['total_high_similarity_pairs'] += 1
                        
                        results['max_similarity'] = max(results['max_similarity'], similarity)
                
                # 添加問題結果
                results['questions'].append({
                    'title': question_col,
                    'heatmap': heatmap_b64,
                    'high_similarity_pairs': high_similarity_pairs,
                    'max_similarity': max([max(row) for row in similarity_matrix])
                })
                
                print(f"   高相似度對: {len(high_similarity_pairs)}")
                print(f"   最高相似度: {max([max(row) for row in similarity_matrix]):.3f}")
            
            else:
                print(f"   ❌ 分析失敗")
                results['questions'].append({
                    'title': question_col,
                    'heatmap': '',
                    'high_similarity_pairs': [],
                    'max_similarity': 0.0
                })
        
        # 生成HTML報告
        output_dir = project_root / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / "comprehensive_plagiarism_report.html"
        generate_html_report(results, report_path)
        
        print(f"\n📊 測試完成總結:")
        print(f"   學生總數: {results['total_students']}")
        print(f"   問題總數: {results['total_questions']}")
        print(f"   高相似度對總數: {results['total_high_similarity_pairs']}")
        print(f"   最高相似度: {results['max_similarity']:.1%}")
        print(f"   報告位置: {report_path}")
        
        print(f"\n🎉 完整抄襲檢測測試成功完成！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
