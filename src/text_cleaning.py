#文本清洗
import os
import re
import codecs
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer



# 初始化工具
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # 1. 移除HTML标签
    text = BeautifulSoup(text, "html.parser").get_text()
    # 2. 去除非字母数字字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. 转换为小写
    text = text.lower()
    # 4. 分词
    words = nltk.word_tokenize(text)
    # 5. 去停用词 & 词干化
    cleaned_words = [
        stemmer.stem(word)  # 或 lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(cleaned_words)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            output_category = os.path.join(output_dir, category)
            os.makedirs(output_category, exist_ok=True)
            
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                # 处理编码问题（常见于新闻数据集）
                with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                cleaned_content = clean_text(content)
                # 保存清洗后的文件
                output_path = os.path.join(output_category, file)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)

# 使用绝对路径避免路径错误！
raw_train_dir = r"D:\MLExp\text-classification-lab\data\20news-bydate-train"
raw_test_dir = r"D:\MLExp\text-classification-lab\data\20news-bydate-test"
cleaned_train_dir = r"D:\MLExp\text-classification-lab\data\cleaned_train"
cleaned_test_dir = r"D:\MLExp\text-classification-lab\data\cleaned_test"

process_directory(raw_train_dir, cleaned_train_dir)
process_directory(raw_test_dir, cleaned_test_dir)