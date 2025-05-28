import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'data/train_augmented.csv'
PLOTS_DIR = 'models/plots/data_analysis2'
os.makedirs(PLOTS_DIR, exist_ok=True)

# データ読み込み
df = pd.read_csv(DATA_PATH, header=0)

# データフレームの構造確認
print("カラム名:", df.columns.tolist())
print("\nデータフレームの最初の5行:")
print(df.head())

# 特徴量とターゲットのカラム名を定義
feature_cols = ['param1', 'param2', 'param3', 'param4', 'param5', 'param6']
target_col = 'class'

# ラベルエンコーディング
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df[target_col])

# 1. 特徴量分布のヒストグラム
for col in feature_cols:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/{col}_hist.png')
    plt.close()

# 2. クラスバランスの棒グラフ
plt.figure()
sns.countplot(x=target_col, data=df)
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
    plt.title(f'{col} vs Class')
    plt.xlabel('Class Label')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/{col}_boxplot.png')
    plt.close()

print(f"データ分析の可視化画像を {PLOTS_DIR} に保存しました。") 
