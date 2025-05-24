import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'data/train.csv'
PLOTS_DIR = 'models/plots/data_analysis'
os.makedirs(PLOTS_DIR, exist_ok=True)

# データ読み込み
df = pd.read_csv(DATA_PATH, header=None)
feature_cols = [0, 1, 2, 3, 4, 5]
target_col = 6

# ラベルエンコーディング
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df[target_col])

# 1. 特徴量分布のヒストグラム
for col in feature_cols:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Feature {col+1} Distribution')
    plt.xlabel(f'param{col+1}')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/feature{col+1}_hist.png')
    plt.close()

# 2. クラスバランスの棒グラフ
plt.figure()
sns.countplot(x=df[target_col], data=df)
plt.title('Class Balance')
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/class_balance.png')
plt.close()

# 3. 特徴量間の相関ヒートマップ
plt.figure(figsize=(8, 6))
corr = df[feature_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/feature_corr_heatmap.png')
plt.close()

# 4. 各特徴量とラベルの関係（boxplot）
for col in feature_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=target_col, y=col, data=df)
    plt.title(f'Feature {col+1} vs Class')
    plt.xlabel('Class Label')
    plt.ylabel(f'param{col+1}')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/feature{col+1}_boxplot.png')
    plt.close()

print(f"データ分析の可視化画像を {PLOTS_DIR} に保存しました。") 
