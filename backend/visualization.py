"""
視覺化模組 - 負責生成相似度矩陣和HTML報告
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from typing import List, Optional, Tuple, Dict, Any
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# 設定中文字體支援
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class VisualizationEngine:
    """視覺化引擎類別"""
    
    def __init__(self):
        """初始化視覺化引擎"""
        self.logger = logging.getLogger(__name__)
    
    async def create_similarity_matrices(self, texts: List[str], names: List[str] = None, 
                                       question_id: int = 1, use_genai: bool = True) -> Dict[str, Optional[str]]:
        """創建相似度矩陣視覺化
        
        Args:
            texts: 文本列表
            names: 姓名列表
            question_id: 問題編號
            use_genai: 是否使用GenAI
            
        Returns:
            Dict: 包含base64編碼圖像的字典
        """
        if len(texts) < 2:
            return {'genai_matrix': None, 'local_matrix': None}
        
        result = {}
        
        # 準備標籤
        labels = self._prepare_labels(names, len(texts))
        
        # 創建GenAI相似度矩陣（如果可用）
        if use_genai:
            try:
                genai_matrix = await self._create_genai_matrix(texts, labels, question_id)
                result['genai_matrix'] = genai_matrix
            except Exception as e:
                self.logger.error(f"GenAI矩陣創建失敗: {e}")
                result['genai_matrix'] = None
        else:
            result['genai_matrix'] = None
        
        # 創建本地相似度矩陣
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
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                    .summary {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                    .question-section {{ margin-bottom: 30px; border: 1px solid #dee2e6; 
                                        border-radius: 8px; padding: 15px; }}
                    .matrix-container {{ text-align: center; margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .high-similarity {{ background-color: #ffebee; }}
                    .medium-similarity {{ background-color: #fff3e0; }}
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
                
                {content_sections}
                
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
            
            similarity_columns = [col for col in df.columns if '_相似度標記' in col]
            similarity_issues = 0
            for col in similarity_columns:
                similarity_issues += sum(1 for flag in df[col] if flag > 0)
            
            # 生成問題部分
            content_sections = ""
            for question_id, viz_data in visualizations.items():
                section = self._generate_question_section(question_id, viz_data)
                content_sections += section
            
            # 生成數據表
            data_table = df.to_html(classes='table table-striped', escape=False)
            
            # 組合最終HTML
            html_content = html_template.format(
                timestamp=timestamp,
                total_students=total_students,
                analyzed_questions=analyzed_questions,
                similarity_issues=similarity_issues,
                content_sections=content_sections,
                data_table=data_table
            )
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"HTML報告生成失敗: {e}")
            return "<html><body><h1>報告生成失敗</h1></body></html>"
    
    def _generate_question_section(self, question_id: str, viz_data: Dict[str, Any]) -> str:
        """生成問題部分的HTML"""
        section = f"""
        <div class="question-section">
            <h3>📝 {question_id} 分析結果</h3>
        """
        
        if viz_data.get('genai_matrix'):
            section += f"""
            <div class="matrix-container">
                <h4>GenAI 語義相似度矩陣</h4>
                <img src="data:image/png;base64,{viz_data['genai_matrix']}" style="max-width: 100%; height: auto;">
            </div>
            """
        
        if viz_data.get('local_matrix'):
            section += f"""
            <div class="matrix-container">
                <h4>多算法相似度矩陣</h4>
                <img src="data:image/png;base64,{viz_data['local_matrix']}" style="max-width: 100%; height: auto;">
            </div>
            """
        
        section += "</div>"
        return section


