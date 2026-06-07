#这段代码的作用是​​统计指定目录下所有子文件夹中的文件总数​
import os

def count_docs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径 {path} 不存在！")
    total = 0
    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if os.path.isdir(category_path):
            total += len(os.listdir(category_path))
    return total

# 调用示例

train_dir = r"D:\MLExp\text-classification-lab\data\20news-bydate-train"
test_dir = r"D:\MLExp\text-classification-lab\data\20news-bydate-test"

# 检查路径是否存在
assert os.path.exists(train_dir), f"路径 {train_dir} 不存在！"
assert os.path.exists(test_dir), f"路径 {test_dir} 不存在！"

print("训练集文档数:", count_docs(train_dir))
print("测试集文档数:", count_docs(test_dir))