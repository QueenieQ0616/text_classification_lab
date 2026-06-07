#代码读取一篇文档
import os

# 数据集路径（根据实际路径修改）
train_dir = r"D:\MLExp\text-classification-lab\data\20news-bydate-train"
test_dir = r"D:\MLExp\text-classification-lab\data\20news-bydate-test"

# 读取训练集第一个类别的第一篇文档
first_category = os.listdir(train_dir)[0]
first_file = os.listdir(os.path.join(train_dir, first_category))[0]
file_path = os.path.join(train_dir, first_category, first_file)

with open(file_path, 'r', encoding='latin1') as f:
    content = f.read()
    print("文档内容（前500字符）:\n", content[:500])