import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

# ------------------------ 路径配置 ------------------------
BASE_DIR = Path(__file__).parents[2]  # 项目根目录
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs/analysis_results"

# ------------------------ 数据加载函数（修复未定义问题） ------------------------
def load_data(data_path):
    texts, labels = [], []
    try:
        for category in os.listdir(data_path):
            cat_path = os.path.join(data_path, category)
            if os.path.isdir(cat_path):
                for fname in os.listdir(cat_path):
                    filepath = os.path.join(cat_path, fname)
                    with open(filepath, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                        labels.append(category)
        return texts, labels
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return [], []

# ------------------------ 主流程（修复路径拼写错误） ------------------------
if __name__ == "__main__":
    # 加载数据
    train_texts, train_labels = load_data(DATA_DIR / "cleaned_train")  # 注意路径名修正

# ------------------------ 特征工程 ------------------------
vectorizer = joblib.load(BASE_DIR / "outputs/naive_bayes_vectorizer.pkl")
X_train = vectorizer.transform(train_texts)

# ------------------------ 超参数网格搜索 ------------------------
param_grid = {
    "alpha": [0.1, 0.5, 1.0, 2.0],
    "fit_prior": [True, False]
}

grid = GridSearchCV(
    MultinomialNB(),
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro"
)
grid.fit(X_train, train_labels)

# ------------------------ 结果分析 ------------------------
results_df = pd.DataFrame(grid.cv_results_)
results_df.to_csv(OUTPUT_DIR / "naive_bayes_hyperparam_results.csv")

# 折线图：alpha 敏感性
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=results_df,
    x="param_alpha",
    y="mean_test_score",
    hue="param_fit_prior",
    marker="o"
)
plt.xscale("log")
plt.title("Naive Bayes Alpha Sensitivity")
plt.savefig(OUTPUT_DIR / "naive_bayes_alpha_sensitivity.png")