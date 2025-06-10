#!/usr/bin/env python3
"""
修改測試檔案以創建抄襲檢測的測試案例
"""

import csv
import sys
import os

def modify_plagiarism_test():
    """修改第三個學生的答案使其與第一個學生相似"""
    
    input_file = '/Users/yangziliang/Git-Projects/AI-TA-Grader/testfile/Final Exam Quiz Student Analysis Report_Public_plag.csv'
    
    # 讀取檔案
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 第一個學生的答案 (Kujo Jotaro)
    original_answer_q1 = '''Comparison of the two methods of Czochralski (CZ) process and  Floating Zone (FZ) process  for single -crystal silicon growth based on advantages and disadvantages as follow below:   
CZ method is more popular used for mass production  for Silicon crystal growth was discovered accidently in 1956.  
1. Cheaper
2. Larger wafer size (300 mm in production)
3. Reusable materials
• Floating Zone process is alternative way to make single-crystal ingot.
1. Pure silicon crystal (no crucible)
2. More expensive, smaller wafer size (150 mm)
3. Mainly for power devices

Even though FZ method is more expensive its useful in making  smaller wafer size typical 150 nm is primary advantage of FZ is higher Si purity with less crucible contamination.

Answer 2. Channeling effect is phenomena that occur during the incident angle is right, ion can travel long distance without collision with lattice atoms. Also the projected range always follows Gaussian distribution. 
 There are many "channels" along certain angles'''
    
    # 修改第三個學生的答案，讓它與第一個學生非常相似但稍有不同
    modified_answer_q1 = '''A1.
Comparison between the Czochralski (CZ) process and Floating Zone (FZ) process for single-crystal silicon growth based on advantages and disadvantages as follow below:
CZ method is more popular used for mass production for Silicon crystal growth was discovered accidently in 1956.
1. Cheaper
2. Larger wafer size (300 mm in production)
3. Reusable materials
• Floating Zone process is alternative way to make single-crystal ingot.
1. Pure silicon crystal (no crucible)
2. More expensive, smaller wafer size (150 mm)  
3. Mainly for power devices

Even though FZ method is more expensive its useful in making smaller wafer size typical 150 nm is primary advantage of FZ is higher Si purity with less crucible contamination.

A2.
Channeling effect is phenomena that occur during the incident angle is right, ion can travel long distance without collision with lattice atoms. Also the projected range always follows Gaussian distribution.
There are many "channels" along certain angles'''
    
    # 替換第三個學生的答案
    # 先找到第三個學生的原始答案並替換
    old_giorno_start = '"A1.\nCZ method is more popular'
    old_giorno_end = '3.Rotate wafer and post-implantation diffusion"'
    
    # 尋找並替換
    start_pos = content.find(old_giorno_start)
    end_pos = content.find(old_giorno_end) + len(old_giorno_end)
    
    if start_pos != -1 and end_pos != -1:
        new_content = content[:start_pos] + '"' + modified_answer_q1 + '\nIt causes uncontrollable dopant profile\nThe most common method to reduce or avoid channeling effect as follow below:\n1. Tilt wafer, 7° is most commonly used\n2. Pre-amorphous implantation by Germanium\n3. Rotate wafer and post-implantation diffusion"' + content[end_pos:]
        
        # 寫回檔案
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ 成功修改測試檔案，第三個學生的答案現在與第一個學生高度相似")
    else:
        print("❌ 找不到要修改的內容")

if __name__ == "__main__":
    modify_plagiarism_test()
