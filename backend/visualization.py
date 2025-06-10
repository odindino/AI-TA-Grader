"""
視覺化模組 - 負責生成相似度矩陣和HTML報告
"""

import logging
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import base64
from typing import List, Optional, Tuple, Dict, Any
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# 設定 matplotlib 後端和中文字體支援
import matplotlib
matplotlib.use('Agg')  # 使用非GUI後端，避免線程問題
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class VisualizationEngine:
    """視覺化引擎類別"""
    
    def __init__(self):
        """初始化視覺化引擎"""
        self.logger = logging.getLogger(__name__)
    
    async def create_similarity_matrices(self, texts: List[str], names: List[str] = None, 
                                       question_id: int = 1, use_genai: bool = True) -> Dict[str, Optional[str]]:
        """創建相似度矩陣視覺化（僅本地算法）
        
        Args:
            texts: 文本列表
            names: 姓名列表
            question_id: 問題編號
            use_genai: 是否使用GenAI（已棄用，僅保留本地算法）
            
        Returns:
            Dict: 包含base64編碼圖像的字典
        """
        if len(texts) < 2:
            return {'local_matrix': None}
        
        result = {}
        
        # 準備標籤
        labels = self._prepare_labels(names, len(texts))
        
        # 只創建本地相似度矩陣
        try:
            local_matrix = self._create_local_matrix(texts, labels, question_id)
            result['local_matrix'] = local_matrix
        except Exception as e:
            self.logger.error(f"本地矩陣創建失敗: {e}")
            result['local_matrix'] = None
        
        return result
    
    def _prepare_labels(self, names: Optional[List[str]], num_texts: int) -> List[str]:
        """準備標籤"""
        if names:
            # 截短標籤以適應顯示
            labels = [name[:8] + '...' if len(name) > 8 else name for name in names]
        else:
            labels = [f"學生{i+1}" for i in range(num_texts)]
        
        return labels
    
    async def _create_genai_matrix(self, texts: List[str], labels: List[str], question_id: int) -> Optional[str]:
        """創建GenAI相似度矩陣"""
        try:
            # 預處理文本
            processed_texts = [t.strip() if t and t.strip() else " " for t in texts]
            
            # 計算GenAI相似度矩陣
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=processed_texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            embs = np.array(result['embedding'])
            genai_similarity_matrix = cosine_similarity(embs)
            
            # 創建視覺化
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
            matrix_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return matrix_b64
            
        except Exception as e:
            self.logger.error(f"GenAI相似度矩陣視覺化失敗: {e}")
            return None
    
    def _create_local_matrix(self, texts: List[str], labels: List[str], question_id: int) -> Optional[str]:
        """創建本地相似度矩陣"""
        try:
            from .alternative_similarity_methods import calculate_advanced_similarity
            
            # 計算本地相似度矩陣
            n = len(texts)
            local_similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j and texts[i].strip() and texts[j].strip():
                        similarity = calculate_advanced_similarity(texts[i], texts[j])
                        local_similarity_matrix[i][j] = similarity
                    elif i == j:
                        local_similarity_matrix[i][j] = 1.0
            
            # 創建本地矩陣視覺化（根據學生數量調整尺寸）
            n = len(texts)
            if n > 20:
                fig_size = (max(12, n * 0.4), max(10, n * 0.4))
                font_size = 6
                rotation_angle = 90
                annot = n <= 25  # 太多學生時不顯示數字
            elif n > 15:
                fig_size = (12, 10)
                font_size = 8
                rotation_angle = 60
                annot = True
            else:
                fig_size = (10, 8)
                font_size = 10
                rotation_angle = 45
                annot = True
            
            plt.figure(figsize=fig_size)
            mask = np.triu(np.ones_like(local_similarity_matrix, dtype=bool))  # 只顯示下三角
            
            # 設定 heatmap 參數
            heatmap_kwargs = {
                'mask': mask,
                'annot': annot,
                'fmt': '.2f' if n > 15 else '.3f',
                'cmap': 'Blues',
                'xticklabels': labels,
                'yticklabels': labels,
                'cbar_kws': {'label': '相似度'},
                'square': True
            }
            
            if annot:
                heatmap_kwargs['annot_kws'] = {'size': font_size}
            
            sns.heatmap(local_similarity_matrix, **heatmap_kwargs)
            
            plt.title(f'Q{question_id} - 相似度分析矩陣 ({n}位學生)', fontsize=14, pad=20)
            plt.xlabel('學生', fontsize=12)
            plt.ylabel('學生', fontsize=12)
            plt.xticks(rotation=rotation_angle, ha='right', fontsize=font_size)
            plt.yticks(rotation=0, fontsize=font_size)
            
            # 調整布局以適應不同尺寸
            if n > 20:
                plt.subplots_adjust(bottom=0.2, left=0.15)
            else:
                plt.tight_layout()
            
            # 轉為base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            matrix_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return matrix_b64
            
        except Exception as e:
            self.logger.error(f"本地相似度矩陣視覺化失敗: {e}")
            return None
    
    async def generate_enhanced_html_report(self, df: pd.DataFrame, visualizations: Dict[str, Any]) -> str:
        """生成增強的HTML報告
        
        Args:
            df: 結果數據框
            visualizations: 視覺化結果字典
            
        Returns:
            str: HTML報告內容
        """
        try:
            # HTML模板
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>AI-TA-Grader 分析報告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                    .summary {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                    .question-section {{ margin-bottom: 30px; border: 1px solid #dee2e6; 
                                        border-radius: 8px; padding: 15px; background: white; }}
                    .matrix-container {{ text-align: center; margin: 20px 0; }}
                    .matrix-container img {{ box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; background: white; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; font-weight: bold; }}
                    .high-similarity {{ background-color: #ffebee; }}
                    .medium-similarity {{ background-color: #fff3e0; }}
                    .ai-high {{ background-color: #ffcdd2; }}
                    .ai-medium {{ background-color: #ffe0b2; }}
                    .ai-low {{ background-color: #c8e6c9; }}
                    .score-table {{ margin: 20px 0; }}
                    .score-table th {{ background-color: #e3f2fd; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 AI-TA-Grader 分析報告</h1>
                    <p>自動化考卷分析與相似度檢測報告</p>
                    <p>生成時間: {timestamp}</p>
                </div>
                
                <div class="summary">
                    <h2>📊 分析摘要</h2>
                    <p><strong>學生總數:</strong> {total_students}</p>
                    <p><strong>分析問題:</strong> {analyzed_questions}</p>
                    <p><strong>檢測到相似度異常:</strong> {similarity_issues}</p>
                </div>
                
                {ai_risk_table}
                
                {content_sections}
                
                <div class="summary">
                    <h2>📊 相似度詳細分析表</h2>
                    {similarity_table}
                </div>
                
                <div class="summary">
                    <h2>📈 相似度矩陣數據表</h2>
                    {similarity_matrix_table}
                </div>
                
                <div class="summary">
                    <h2>🤖 AI 分析詳細記錄</h2>
                    {ai_logs}
                </div>
                
                <div class="summary">
                    <h2>📋 完整數據表</h2>
                    {data_table}
                </div>
            </body>
            </html>
            """
            
            # 生成內容
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 生成摘要信息
            total_students = len(df)
            score_columns = [col for col in df.columns if '_分數' in col]
            analyzed_questions = ', '.join([col.replace('_分數', '') for col in score_columns])
            
            similarity_columns = [col for col in df.columns if '_相似度分數' in col]
            similarity_issues = 0
            for col in similarity_columns:
                similarity_issues += sum(1 for score in df[col] if pd.notna(score) and score > 70)
            
            # 生成AI風險表格
            ai_risk_table = self._generate_ai_risk_table(df)
            
            # 生成問題部分
            content_sections = ""
            for question_id, viz_data in visualizations.items():
                section = self._generate_question_section(question_id, viz_data)
                content_sections += section
            
            # 生成AI記錄部分
            ai_logs = self._generate_ai_logs_section()
            
            # 生成相似度表格
            similarity_table = self._generate_similarity_table(df)
            
            # 生成相似度矩陣表格
            similarity_matrix_table = self._generate_similarity_matrix_table(visualizations, df)
            
            # 生成數據表 - 只顯示重要欄位
            important_cols = ['name'] + [col for col in df.columns if '_分數' in col or '_相似度分數' in col or '_AI風險' in col]
            display_df = df[important_cols] if all(col in df.columns for col in important_cols) else df
            data_table = display_df.to_html(classes='table table-striped', escape=False, index=False)
            
            # 組合最終HTML
            html_content = html_template.format(
                timestamp=timestamp,
                total_students=total_students,
                analyzed_questions=analyzed_questions,
                similarity_issues=similarity_issues,
                ai_risk_table=ai_risk_table,
                content_sections=content_sections,
                similarity_table=similarity_table,
                similarity_matrix_table=similarity_matrix_table,
                ai_logs=ai_logs,
                data_table=data_table
            )
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"HTML報告生成失敗: {e}")
            return "<html><body><h1>報告生成失敗</h1></body></html>"
    
    def _generate_question_section(self, question_id: str, viz_data: Dict[str, Any]) -> str:
        """生成問題部分的HTML"""
        try:
            section_html = f"""
            <div class="question-section">
                <h2>📝 {question_id} 分析結果</h2>
            """
            
            # 添加相似度矩陣圖片（僅本地算法）
            if viz_data:
                if 'local_matrix' in viz_data and viz_data['local_matrix']:
                    section_html += f"""
                    <div class="matrix-container">
                        <h3>相似度分析矩陣</h3>
                        <img src="data:image/png;base64,{viz_data['local_matrix']}" style="max-width: 100%; height: auto;">
                    </div>
                    """
            
            section_html += "</div>"
            return section_html
            
        except Exception as e:
            self.logger.error(f"生成問題部分失敗: {e}")
            return ""
    
    def _generate_ai_risk_table(self, df: pd.DataFrame) -> str:
        """生成AI使用嫌疑度表格"""
        try:
            # 查找所有AI風險欄位
            ai_risk_columns = [col for col in df.columns if 'AI風險' in col or 'ai_risk' in col]
            
            if not ai_risk_columns:
                return ""
            
            # 建立AI風險表格
            table_html = """
            <div class="summary">
                <h2>🤖 AI使用嫌疑度分析</h2>
                <table class="score-table">
                    <thead>
                        <tr>
                            <th>學生姓名</th>
            """
            
            # 添加問題欄位
            for col in ai_risk_columns:
                q_num = col.replace('Q', '').replace('_AI風險', '').replace('_ai_risk', '')
                table_html += f"<th>Q{q_num} AI風險</th>"
            
            table_html += """
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # 添加每個學生的數據
            for idx, row in df.iterrows():
                table_html += "<tr>"
                table_html += f"<td>{row.get('name', f'學生{idx+1}')}</td>"
                
                for col in ai_risk_columns:
                    risk_value = row.get(col, 0)
                    if pd.isna(risk_value):
                        risk_value = 0
                    
                    # 根據風險值設定顏色
                    if risk_value >= 70:
                        css_class = "ai-high"
                    elif risk_value >= 40:
                        css_class = "ai-medium"
                    else:
                        css_class = "ai-low"
                    
                    table_html += f'<td class="{css_class}">{int(risk_value)}</td>'
                
                table_html += "</tr>"
            
            table_html += """
                    </tbody>
                </table>
                <p style="margin-top: 10px; font-size: 0.9em;">
                    <span style="background: #ffcdd2; padding: 2px 8px;">高風險 (≥70)</span>
                    <span style="background: #ffe0b2; padding: 2px 8px;">中風險 (40-69)</span>
                    <span style="background: #c8e6c9; padding: 2px 8px;">低風險 (<40)</span>
                </p>
            </div>
            """
            
            return table_html
            
        except Exception as e:
            self.logger.error(f"生成AI風險表格失敗: {e}")
            return ""
    
    def _generate_ai_logs_section(self) -> str:
        """生成AI分析記錄部分"""
        return """
        <p><strong>注意：</strong> AI 分析的詳細記錄會顯示在控制台輸出中，包括：</p>
        <ul>
            <li>🤖 發送給 Gemini 的 Prompt 內容</li>
            <li>📝 學生回答摘要</li>
            <li>🤖 Gemini 的完整回應</li>
            <li>📊 解析出的分數和 AI 風險評估</li>
            <li>🔍 相似度檢測詳細結果</li>
        </ul>
        <p><em>如果您使用了有效的 API 金鑰，這些記錄會即時顯示在運行程式的終端視窗中。</em></p>
        """
    
    def _generate_similarity_table(self, df: pd.DataFrame) -> str:
        """生成相似度分析表格"""
        try:
            # 查找所有相似度分數欄位
            similarity_columns = [col for col in df.columns if '_相似度分數' in col]
            
            if not similarity_columns:
                return "<p>無相似度分析數據</p>"
            
            # 建立相似度表格
            table_html = """
            <table class="score-table" style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #e3f2fd;">
                        <th style="border: 1px solid #ddd; padding: 8px;">學生姓名</th>
            """
            
            # 添加問題欄位
            for col in similarity_columns:
                q_num = col.replace('Q', '').replace('_相似度分數', '')
                table_html += f'<th style="border: 1px solid #ddd; padding: 8px;">Q{q_num} 相似度</th>'
            
            table_html += """
                    </tr>
                </thead>
                <tbody>
            """
            
            # 添加每個學生的數據
            for idx, row in df.iterrows():
                table_html += "<tr>"
                table_html += f'<td style="border: 1px solid #ddd; padding: 8px;">{row.get("name", f"學生{idx+1}")}</td>'
                
                for col in similarity_columns:
                    score = row.get(col, 0)
                    if pd.isna(score):
                        score = 0
                    
                    # 根據相似度分數設定顏色
                    if score >= 85:
                        css_style = "background-color: #ffcdd2; border: 1px solid #ddd; padding: 8px;"  # 高相似度-紅色
                    elif score >= 70:
                        css_style = "background-color: #ffe0b2; border: 1px solid #ddd; padding: 8px;"  # 中相似度-橘色
                    elif score >= 50:
                        css_style = "background-color: #fff3e0; border: 1px solid #ddd; padding: 8px;"  # 低相似度-黃色
                    else:
                        css_style = "background-color: #c8e6c9; border: 1px solid #ddd; padding: 8px;"  # 無相似度-綠色
                    
                    table_html += f'<td style="{css_style}">{int(score)}</td>'
                
                table_html += "</tr>"
            
            table_html += """
                    </tbody>
                </table>
                <p style="margin-top: 10px; font-size: 0.9em;">
                    <span style="background: #ffcdd2; padding: 2px 8px; margin-right: 10px;">高相似度 (≥85分)</span>
                    <span style="background: #ffe0b2; padding: 2px 8px; margin-right: 10px;">中相似度 (70-84分)</span>
                    <span style="background: #fff3e0; padding: 2px 8px; margin-right: 10px;">低相似度 (50-69分)</span>
                    <span style="background: #c8e6c9; padding: 2px 8px;">無相似度 (<50分)</span>
                </p>
            """
            
            return table_html
            
        except Exception as e:
            self.logger.error(f"生成相似度表格失敗: {e}")
            return "<p>相似度表格生成失敗</p>"
    
    def _generate_similarity_matrix_table(self, visualizations: Dict[str, Any], df: pd.DataFrame) -> str:
        """生成相似度矩陣數據表"""
        try:
            if not visualizations:
                return "<p>無相似度矩陣數據</p>"
            
            table_html = ""
            
            for question_id, viz_data in visualizations.items():
                if 'matrix_data' in viz_data:
                    matrix = viz_data['matrix_data']
                    names = viz_data.get('names', [])
                    
                    table_html += f"""
                    <h3>{question_id} 相似度矩陣數據</h3>
                    <div style="overflow-x: auto; margin-bottom: 30px;">
                        <table style="border-collapse: collapse; margin: 10px 0;">
                            <thead>
                                <tr style="background-color: #f0f0f0;">
                                    <th style="border: 1px solid #ddd; padding: 4px; font-size: 12px;">學生</th>
                    """
                    
                    # 添加列標題
                    for i, name in enumerate(names):
                        short_name = name[:6] + "..." if len(name) > 6 else name
                        table_html += f'<th style="border: 1px solid #ddd; padding: 4px; font-size: 10px; min-width: 50px;">{short_name}</th>'
                    
                    table_html += "</tr></thead><tbody>"
                    
                    # 添加數據行
                    for i, name in enumerate(names):
                        table_html += "<tr>"
                        short_name = name[:6] + "..." if len(name) > 6 else name
                        table_html += f'<td style="border: 1px solid #ddd; padding: 4px; font-size: 10px; background-color: #f9f9f9;">{short_name}</td>'
                        
                        for j in range(len(names)):
                            if i == j:
                                # 對角線
                                table_html += '<td style="border: 1px solid #ddd; padding: 4px; text-align: center; background-color: #e0e0e0; font-size: 10px;">1.00</td>'
                            elif i > j:
                                # 下三角（顯示數值）
                                value = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0
                                color = self._get_similarity_color(value)
                                table_html += f'<td style="border: 1px solid #ddd; padding: 4px; text-align: center; background-color: {color}; font-size: 10px;">{value:.3f}</td>'
                            else:
                                # 上三角（留空）
                                table_html += '<td style="border: 1px solid #ddd; padding: 4px; background-color: #f5f5f5;"></td>'
                        
                        table_html += "</tr>"
                    
                    table_html += "</tbody></table></div>"
            
            if not table_html:
                return "<p>無可用的相似度矩陣數據</p>"
            
            # 添加顏色說明
            table_html += """
            <p style="margin-top: 10px; font-size: 0.9em;">
                <strong>相似度範圍：</strong>
                <span style="background: #ff9999; padding: 2px 8px; margin-right: 10px;">0.8-1.0 (高)</span>
                <span style="background: #ffcc99; padding: 2px 8px; margin-right: 10px;">0.5-0.8 (中)</span>
                <span style="background: #ffffcc; padding: 2px 8px; margin-right: 10px;">0.3-0.5 (低)</span>
                <span style="background: #ccffcc; padding: 2px 8px;">&lt;0.3 (無)</span>
            </p>
            """
            
            return table_html
            
        except Exception as e:
            self.logger.error(f"生成相似度矩陣表格失敗: {e}")
            return "<p>相似度矩陣表格生成失敗</p>"
    
    def _get_similarity_color(self, value: float) -> str:
        """根據相似度值取得顏色"""
        if value >= 0.8:
            return '#ff9999'  # 高相似度 - 淺紅
        elif value >= 0.5:
            return '#ffcc99'  # 中相似度 - 淺橘
        elif value >= 0.3:
            return '#ffffcc'  # 低相似度 - 淺黃
        else:
            return '#ccffcc'  # 無相似度 - 淺綠


