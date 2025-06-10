#!/usr/bin/env python3
"""
專門測試抄襲檢測功能的腳本 - 不使用API密鑰
僅使用本地非GenAI方法進行相似度檢測

測試目標：
1. 使用後端模組測試抄襲檢測功能
2. 不依賴任何API密鑰
3. 生成視覺化相似度矩陣和HTML報告
4. 測試修改後的抄襲檔案 (_plag.csv)
"""

import os
import sys
import time
import logging
from pathlib import Path

# 添加項目根目錄到Python路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.analyzer import AnalysisEngine
    from backend.data_processor import DataProcessor
    from backend.similarity_detector import SimilarityDetector
    from backend.visualization import VisualizationEngine
    from backend.config import Config
except ImportError as e:
    print(f"❌ 模組匯入失敗: {e}")
    print(f"請確認您位於項目根目錄: {project_root}")
    sys.exit(1)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlagiarismTesterNoAPI:
    """不使用API的抄襲檢測測試器"""
    
    def __init__(self):
        """初始化測試器"""
        self.project_root = project_root
        self.test_data_path = self.project_root / "testfile"
        self.output_dir = self.project_root / "test_output"
        
        # 確保輸出目錄存在
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化後端組件（不使用API）
        self.config = Config()
        self.data_processor = DataProcessor()
        self.similarity_detector = SimilarityDetector()
        self.visualization = VisualizationEngine()
        
        # 初始化分析引擎（離線模式）
        self.analysis_engine = AnalysisEngine(api_key=None)
        
        logger.info("✅ 抄襲檢測測試器初始化完成（離線模式）")
    
    def test_file_availability(self):
        """測試檔案可用性"""
        logger.info("📂 檢查測試檔案...")
        
        original_file = self.test_data_path / "Final Exam Quiz Student Analysis Report_Public.csv"
        plagiarism_file = self.test_data_path / "Final Exam Quiz Student Analysis Report_Public_plag.csv"
        
        files_status = {
            "原始檔案": original_file.exists(),
            "抄襲測試檔案": plagiarism_file.exists()
        }
        
        for file_name, exists in files_status.items():
            status = "✅" if exists else "❌"
            logger.info(f"{status} {file_name}: {exists}")
        
        return all(files_status.values())
    
    def load_test_data(self, file_path):
        """載入測試資料"""
        logger.info(f"📊 載入測試資料: {file_path.name}")
        
        try:
            # 使用DataProcessor載入資料
            df = self.data_processor.load_csv(str(file_path))
            logger.info(f"✅ 成功載入 {len(df)} 筆資料")
            
            # 顯示資料概況
            if not df.empty:
                columns = list(df.columns)
                logger.info(f"📋 欄位: {columns}")
                logger.info(f"📈 資料形狀: {df.shape}")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ 載入資料失敗: {e}")
            return None
    
    def test_similarity_detection(self, df, test_name="default"):
        """測試相似度檢測"""
        logger.info(f"🔍 開始相似度檢測測試: {test_name}")
        
        try:
            # 提取問題欄位（排除學生資訊欄位）
            question_columns = [col for col in df.columns 
                              if col not in ['Student', 'Email', 'Timestamp']]
            
            logger.info(f"📝 找到 {len(question_columns)} 個問題欄位")
            
            # 為每個問題進行相似度檢測
            all_similarities = {}
            
            for question in question_columns:
                logger.info(f"🔍 分析問題: {question}")
                
                # 取得該問題的所有回答
                answers = df[question].dropna().astype(str).tolist()
                
                if len(answers) < 2:
                    logger.warning(f"⚠️ 問題 {question} 回答數量不足（{len(answers)}）")
                    continue
                
                # 使用SimilarityDetector進行檢測
                similarities = self.similarity_detector.calculate_similarity_matrix(answers)
                all_similarities[question] = similarities
                
                # 找出高相似度對
                high_similarity_pairs = []
                threshold = 0.7  # 70%相似度閾值
                
                for i in range(len(similarities)):
                    for j in range(i + 1, len(similarities[i])):
                        if similarities[i][j] > threshold:
                            high_similarity_pairs.append({
                                'student1': i,
                                'student2': j,
                                'similarity': similarities[i][j]
                            })
                
                if high_similarity_pairs:
                    logger.info(f"🚨 問題 {question} 發現 {len(high_similarity_pairs)} 對高相似度回答:")
                    for pair in high_similarity_pairs:
                        logger.info(f"   學生 {pair['student1']} vs 學生 {pair['student2']}: {pair['similarity']:.3f}")
                else:
                    logger.info(f"✅ 問題 {question} 未發現可疑相似度")
            
            return all_similarities
        
        except Exception as e:
            logger.error(f"❌ 相似度檢測失敗: {e}")
            return None
    
    def generate_visualization(self, similarities, test_name="default"):
        """生成視覺化結果"""
        logger.info(f"📊 生成視覺化結果: {test_name}")
        
        try:
            # 為每個問題生成相似度矩陣圖
            visualization_results = {}
            
            for question, similarity_matrix in similarities.items():
                logger.info(f"🎨 生成問題 {question} 的視覺化矩陣")
                
                # 使用VisualizationEngine生成圖表
                matrix_image = self.visualization.create_similarity_matrices({question: similarity_matrix})
                
                if matrix_image:
                    visualization_results[question] = matrix_image
                    logger.info(f"✅ 成功生成問題 {question} 的視覺化")
                else:
                    logger.warning(f"⚠️ 生成問題 {question} 視覺化失敗")
            
            return visualization_results
        
        except Exception as e:
            logger.error(f"❌ 視覺化生成失敗: {e}")
            return None
    
    def generate_html_report(self, df, similarities, visualizations, test_name="default"):
        """生成HTML報告"""
        logger.info(f"📄 生成HTML報告: {test_name}")
        
        try:
            # 準備報告資料
            report_data = {
                'df': df,
                'similarities': similarities,
                'visualizations': visualizations,
                'test_info': {
                    'name': test_name,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'mode': '離線模式（無API）',
                    'algorithms': '6種本地相似度算法'
                }
            }
            
            # 使用VisualizationEngine生成HTML報告
            html_content = self.visualization.generate_enhanced_html_report(
                df, similarities, report_data['test_info']
            )
            
            # 儲存HTML報告
            report_path = self.output_dir / f"plagiarism_report_{test_name}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ HTML報告已儲存: {report_path}")
            return report_path
        
        except Exception as e:
            logger.error(f"❌ HTML報告生成失敗: {e}")
            return None
    
    def run_complete_test(self):
        """執行完整的抄襲檢測測試"""
        logger.info("🚀 開始完整抄襲檢測測試")
        logger.info("=" * 60)
        
        # 1. 檢查檔案可用性
        if not self.test_file_availability():
            logger.error("❌ 測試檔案不可用，終止測試")
            return False
        
        # 2. 測試原始檔案
        logger.info("\n📊 測試原始檔案...")
        original_file = self.test_data_path / "Final Exam Quiz Student Analysis Report_Public.csv"
        original_df = self.load_test_data(original_file)
        
        if original_df is not None:
            original_similarities = self.test_similarity_detection(original_df, "original")
            if original_similarities:
                original_viz = self.generate_visualization(original_similarities, "original")
                self.generate_html_report(original_df, original_similarities, original_viz, "original")
        
        # 3. 測試抄襲檔案
        logger.info("\n🚨 測試抄襲檔案...")
        plagiarism_file = self.test_data_path / "Final Exam Quiz Student Analysis Report_Public_plag.csv"
        plagiarism_df = self.load_test_data(plagiarism_file)
        
        if plagiarism_df is not None:
            plagiarism_similarities = self.test_similarity_detection(plagiarism_df, "plagiarism")
            if plagiarism_similarities:
                plagiarism_viz = self.generate_visualization(plagiarism_similarities, "plagiarism")
                self.generate_html_report(plagiarism_df, plagiarism_similarities, plagiarism_viz, "plagiarism")
        
        # 4. 比較結果
        logger.info("\n📊 測試結果摘要:")
        logger.info("=" * 40)
        
        if original_similarities and plagiarism_similarities:
            logger.info("✅ 兩個檔案都成功分析")
            
            # 計算平均相似度
            for test_name, similarities in [("原始檔案", original_similarities), ("抄襲檔案", plagiarism_similarities)]:
                total_similarity = 0
                count = 0
                max_similarity = 0
                
                for question, matrix in similarities.items():
                    for i in range(len(matrix)):
                        for j in range(i + 1, len(matrix[i])):
                            similarity = matrix[i][j]
                            total_similarity += similarity
                            count += 1
                            max_similarity = max(max_similarity, similarity)
                
                if count > 0:
                    avg_similarity = total_similarity / count
                    logger.info(f"📊 {test_name} - 平均相似度: {avg_similarity:.3f}, 最高相似度: {max_similarity:.3f}")
        
        logger.info(f"\n📁 輸出檔案位置: {self.output_dir}")
        logger.info("🎉 完整測試完成！")
        
        return True
    
    def run_focused_plagiarism_test(self):
        """專注於抄襲檢測的測試"""
        logger.info("🎯 專注抄襲檢測測試")
        logger.info("=" * 50)
        
        # 載入抄襲測試檔案
        plagiarism_file = self.test_data_path / "Final Exam Quiz Student Analysis Report_Public_plag.csv"
        
        if not plagiarism_file.exists():
            logger.error(f"❌ 抄襲測試檔案不存在: {plagiarism_file}")
            return False
        
        df = self.load_test_data(plagiarism_file)
        if df is None:
            return False
        
        # 執行相似度檢測
        similarities = self.test_similarity_detection(df, "focused_plagiarism")
        if not similarities:
            return False
        
        # 生成詳細分析
        logger.info("\n🔍 詳細抄襲分析:")
        logger.info("-" * 30)
        
        for question, matrix in similarities.items():
            logger.info(f"\n📝 問題: {question}")
            
            # 找出所有相似度 > 0.5 的對
            suspicious_pairs = []
            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix[i])):
                    similarity = matrix[i][j]
                    if similarity > 0.5:
                        suspicious_pairs.append((i, j, similarity))
            
            if suspicious_pairs:
                suspicious_pairs.sort(key=lambda x: x[2], reverse=True)
                logger.info(f"🚨 發現 {len(suspicious_pairs)} 對可疑相似:")
                
                for idx, (i, j, sim) in enumerate(suspicious_pairs[:5]):  # 顯示前5個
                    logger.info(f"   {idx+1}. 學生 {i} vs 學生 {j}: {sim:.3f}")
                    
                    # 顯示實際回答內容（前100字符）
                    answer1 = str(df.iloc[i][question])[:100]
                    answer2 = str(df.iloc[j][question])[:100]
                    logger.info(f"      學生 {i}: {answer1}...")
                    logger.info(f"      學生 {j}: {answer2}...")
            else:
                logger.info("✅ 未發現可疑相似度")
        
        # 生成視覺化和報告
        visualizations = self.generate_visualization(similarities, "focused_plagiarism")
        html_report = self.generate_html_report(df, similarities, visualizations, "focused_plagiarism")
        
        if html_report:
            logger.info(f"\n📄 詳細報告已生成: {html_report}")
        
        return True


def main():
    """主函數"""
    print("🔍 AI-TA-Grader 抄襲檢測測試（無API模式）")
    print("=" * 60)
    
    # 創建測試器
    tester = PlagiarismTesterNoAPI()
    
    # 選擇測試模式
    print("\n選擇測試模式:")
    print("1. 完整測試（原始檔案 + 抄襲檔案）")
    print("2. 專注抄襲檢測測試")
    print("3. 兩者都執行")
    
    try:
        choice = input("\n請輸入選擇 (1/2/3): ").strip()
        
        if choice == "1":
            tester.run_complete_test()
        elif choice == "2":
            tester.run_focused_plagiarism_test()
        elif choice == "3":
            tester.run_complete_test()
            print("\n" + "="*60)
            tester.run_focused_plagiarism_test()
        else:
            print("❌ 無效選擇，執行專注抄襲檢測測試")
            tester.run_focused_plagiarism_test()
    
    except KeyboardInterrupt:
        print("\n\n👋 測試被用戶中斷")
    except Exception as e:
        logger.error(f"❌ 測試執行失敗: {e}")


if __name__ == "__main__":
    main()
