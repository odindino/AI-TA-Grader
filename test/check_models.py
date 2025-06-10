#!/usr/bin/env python3
# check_models.py
# 檢查 Google Generative AI 可用的模型

import google.generativeai as genai
import os

def list_available_models():
    """列出所有可用的 Gemini 模型"""
    print("=== 檢查 Google Generative AI 可用模型 ===")
    
    # 檢查是否有 API 金鑰
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  請設定 GEMINI_API_KEY 環境變數或在此處輸入 API 金鑰")
        api_key = input("請輸入您的 Gemini API 金鑰: ").strip()
        if not api_key:
            print("❌ 沒有提供 API 金鑰，無法檢查模型")
            return
    
    try:
        # 配置 API
        genai.configure(api_key=api_key)
        print("✅ API 金鑰設定成功\n")
        
        # 列出所有模型
        print("📋 可用的模型列表：")
        print("-" * 60)
        
        generation_models = []
        embedding_models = []
        
        print("正在獲取模型列表...")
        models_list = list(genai.list_models())
        print(f"發現 {len(models_list)} 個模型")
        
        for i, model in enumerate(models_list):
            try:
                print(f"\n處理第 {i+1} 個模型...")
                print(f"模型類型: {type(model)}")
                
                # 嘗試不同的屬性名稱
                if hasattr(model, 'name'):
                    model_name = model.name
                elif hasattr(model, 'model_name'):
                    model_name = model.model_name
                elif isinstance(model, str):
                    model_name = model
                else:
                    print(f"無法取得模型名稱，模型物件: {model}")
                    continue
                
                print(f"模型名稱: {model_name}")
                
                # 檢查支援的方法
                if hasattr(model, 'supported_generation_methods'):
                    supported_methods = [method.name if hasattr(method, 'name') else str(method) 
                                       for method in model.supported_generation_methods]
                    print(f"支援方法: {', '.join(supported_methods)}")
                    
                    if 'generateContent' in supported_methods:
                        generation_models.append(model_name)
                        print("🤖 [適用於文字生成]")
                    
                    if 'embedContent' in supported_methods:
                        embedding_models.append(model_name)
                        print("🔤 [適用於文字嵌入]")
                else:
                    print("無法取得支援的方法資訊")
                    # 如果模型名稱包含 gemini 就假設支援生成
                    if 'gemini' in model_name.lower():
                        generation_models.append(model_name)
                        print("🤖 [假設適用於文字生成]")
                
                print("-" * 60)
                
            except Exception as e:
                print(f"處理模型時發生錯誤: {e}")
                print(f"模型物件: {model}")
                continue
        
        print(f"\n📊 總結:")
        print(f"支援文字生成的模型 ({len(generation_models)} 個):")
        for model in generation_models:
            # 去掉 "models/" 前綴以便使用
            clean_name = model.replace("models/", "")
            print(f"  - {clean_name}")
        
        print(f"\n支援文字嵌入的模型 ({len(embedding_models)} 個):")
        for model in embedding_models:
            clean_name = model.replace("models/", "")
            print(f"  - {clean_name}")
            
        # 推薦模型
        print(f"\n💡 推薦用於分析的模型:")
        recommended = [m for m in generation_models if 'gemini' in m.lower() and 'pro' in m.lower()]
        for model in recommended:
            clean_name = model.replace("models/", "")
            print(f"  - {clean_name}")
        
    except Exception as e:
        print(f"❌ 檢查模型時發生錯誤: {e}")
        print("請檢查您的 API 金鑰是否正確")

if __name__ == "__main__":
    list_available_models()
