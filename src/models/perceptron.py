import os
import joblib
import logging
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PerceptronModel:
    """封装感知机模型的训练、评估和保存逻辑"""
    
    def __init__(self, model_params=None):
        """
        初始化模型
        :param model_params: 模型超参数（字典格式）
        """
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        default_params = {
            "penalty": "l2",
            "alpha": 0.001,
            "max_iter": 100,
            "random_state": 42
        }
        self.params = model_params if model_params else default_params

    def train(self, train_texts, train_labels):
        """训练模型"""
        # 特征提取
        X_train = self.vectorizer.fit_transform(train_texts)
        # 初始化模型
        self.model = Perceptron(**self.params)
        # 训练
        self.model.fit(X_train, train_labels)
        logger.info("Perceptron model training completed.")

    def evaluate(self, test_texts, test_labels):
        """评估模型性能"""
        X_test = self.vectorizer.transform(test_texts)
        y_pred = self.model.predict(X_test)
        # 输出指标
        accuracy = accuracy_score(test_labels, y_pred)
        report = classification_report(test_labels, y_pred)
        logger.info(f"Accuracy: {accuracy:.4f}\nClassification Report:\n{report}")
        return accuracy, report

    def save(self, output_dir):
        """保存模型和特征提取器"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 保存模型
        model_path = os.path.join(output_dir, "perceptron_model.pkl")
        joblib.dump(self.model, model_path)
        # 保存特征提取器
        vectorizer_path = os.path.join(output_dir, "perceptron_vectorizer.pkl")
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info(f"Model and vectorizer saved to {output_dir}")

def load_data(data_path):
    """加载文本数据（假设数据按类别分目录存放）"""
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

if __name__ == "__main__":
    # ----------- 路径配置 -----------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # ----------- 数据加载 -----------
    train_texts, train_labels = load_data(os.path.join(DATA_DIR, "cleaned_train"))
    test_texts, test_labels = load_data(os.path.join(DATA_DIR, "cleaned_test"))
    
    # ----------- 模型训练 -----------
    model = PerceptronModel(model_params={"alpha": 0.01, "max_iter": 200})  # 可自定义参数
    model.train(train_texts, train_labels)
    
    # ----------- 模型评估 -----------
    model.evaluate(test_texts, test_labels)
    
    # ----------- 模型保存 -----------
    model.save(OUTPUT_DIR)