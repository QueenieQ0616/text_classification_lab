import os
import argparse
import joblib
from models.perceptron import PerceptronModel
from models.knn import KNNModel
from models.naive_bayes import NaiveBayesModel

def load_data(data_path):
    """复用各模型模块中的加载函数（需确保一致）"""
    # 注意：实际应统一提取到 utils 模块，此处简化
    texts, labels = [], []
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if os.path.isdir(category_dir):
            for filename in os.listdir(category_dir):
                filepath = os.path.join(category_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(category)
    return texts, labels

def main():
    # ----------- 参数解析 -----------
    parser = argparse.ArgumentParser(description="文本分类模型训练入口")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["perceptron", "knn", "naive_bayes"],
                        help="选择模型：perceptron, knn, naive_bayes")
    parser.add_argument("--train_data", type=str, 
                        default="data/cleaned_train",
                        help="训练数据路径")
    parser.add_argument("--test_data", type=str, 
                        default="data/cleaned_test",
                        help="测试数据路径")
    parser.add_argument("--output_dir", type=str, 
                        default="outputs",
                        help="模型保存目录")
    args = parser.parse_args()

    # ----------- 数据加载 -----------
    train_texts, train_labels = load_data(args.train_data)
    test_texts, test_labels = load_data(args.test_data)

    # ----------- 模型初始化 -----------
    if args.model == "perceptron":
        model = PerceptronModel()
    elif args.model == "knn":
        model = KNNModel()
    elif args.model == "naive_bayes":
        model = NaiveBayesModel()
    else:
        raise ValueError("不支持的模型类型")

    # ----------- 训练与评估 -----------
    model.train(train_texts, train_labels)
    model.evaluate(test_texts, test_labels)
    
    # ----------- 模型保存 -----------
    model.save(args.output_dir)

if __name__ == "__main__":
    main()