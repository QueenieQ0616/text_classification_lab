# 词袋模型特征提取
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import coo_matrix
import numpy as np

# --------------------------------------------
# 1. 加载清洗后的训练数据（修正原代码路径变量）
# --------------------------------------------
cleaned_data_path = r"D:\MLExp\text-classification-lab\data\cleaned_train"  # 请确认实际路径
train_texts = []
train_labels = []

# 遍历清洗后的目录
for category in os.listdir(cleaned_data_path):
    category_path = os.path.join(cleaned_data_path, category)
    if os.path.isdir(category_path):
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                train_texts.append(f.read())
                train_labels.append(category)  # 假设按文件夹名作为标签

# 2. 生成TF-IDF特征矩阵（保留原功能）
vectorizer = TfidfVectorizer(max_features=5000)  # 保持原参数
X_train = vectorizer.fit_transform(train_texts)

# 3. 核心优化：生成符合要求的词典和稀疏表示
# 3.1 生成词典文件（ID从1开始）
vocab_dict = {word: idx+1 for idx, word in enumerate(vectorizer.get_feature_names_out())}  # 关键调整：ID从1开始
with open("vocab-dictionary.json", "w") as f:
    json.dump(vocab_dict, f, indent=2)

# 3.2 生成稀疏表示文件（词项ID+出现次数）
# 使用CountVectorizer获取原始词频（非TF-IDF加权）
count_vectorizer = CountVectorizer(max_features=5000, vocabulary=vectorizer.vocabulary_)
X_count = count_vectorizer.fit_transform(train_texts)

# 将稀疏矩阵转换为COO格式便于遍历
X_coo = X_count.tocoo()

# 按文档分组写入文件
with open("sparse-representations.txt", "w") as f:
    for doc_id in range(X_coo.shape[0]):
        # 提取当前文档的非零词项ID和次数
        mask = X_coo.row == doc_id
        cols = X_coo.col[mask] + 1  # ID从1开始
        counts = X_coo.data[mask]
        
        # 格式化为 <词项ID,次数>
        term_pairs = [f"<{col},{count}>" for col, count in zip(cols, counts)]
        f.write(f"D{doc_id+1}: {', '.join(term_pairs)}\n")

# --------------------------------------------
# 4. 原调试代码（保持原有检查逻辑）
# --------------------------------------------
print("\n[调试信息]")
print("清洗后的训练集路径是否存在:", os.path.exists(cleaned_data_path))  # 应为True
print("首个子文件夹名称示例:", os.listdir(cleaned_data_path)[0])  # 如"sci.space"