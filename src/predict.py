import os
import joblib

# ------------------------ 路径配置 ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ------------------------ 加载模型和向量化器 ------------------------
def load_model_and_vectorizer(model_name):
    """加载指定模型的 PKL 文件"""
    try:
        model_path = os.path.join(OUTPUT_DIR, f"{model_name}_model.pkl")
        vectorizer_path = os.path.join(OUTPUT_DIR, f"{model_name}_vectorizer.pkl")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        print(f"加载失败: {str(e)}")
        return None, None

# ------------------------ 预测函数 ------------------------
def predict(texts, model_name="perceptron"):
    """文本分类预测"""
    # 加载模型和特征提取器
    model, vectorizer = load_model_and_vectorizer(model_name)
    if not model or not vectorizer:
        return []

    # 文本向量化（必须使用对应模型的 vectorizer！）
    X = vectorizer.transform(texts)
    # 预测类别
    predictions = model.predict(X)
    return predictions

# ------------------------ 示例用法 ------------------------
if __name__ == "__main__":
    # 示例文本（支持单条或多条）
    new_texts = [
        "NASA announces new Mars rover mission",  # 预期类别：sci.space
        "Latest GPU technology from NVIDIA",       # 预期类别：comp.graphics
        "Stock market hits all-time high"          # 预期类别：其他类别
    ]

    # 选择模型（可选：perceptron, knn, naive_bayes）
    selected_model = "knn"

    # 执行预测
    results = predict(new_texts, model_name=selected_model)

    # 输出结果
    print("\n预测结果：")
    for text, label in zip(new_texts, results):
        print(f"文本：{text[:50]}... → 类别：{label}")