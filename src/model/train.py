import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.preprocessor import DataPreprocessor
from src.model.trainer import ModelTrainer as EnsembleTrainer
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import logging
from sklearn.preprocessing import LabelEncoder

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(conf_matrix, classes, save_path):
    """混同行列のプロット"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, save_path):
    """特徴量重要度のプロット"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        importances = model.get_booster().get_score(importance_type='weight')
        importances = [importances.get(f'f{i}', 0) for i in range(len(feature_names))]
    else:
        return
    
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(scores: Dict[str, float], save_path: str):
    """モデル間の性能比較をプロット"""
    plt.figure(figsize=(12, 6))
    
    # 個別モデルのスコア
    model_scores = {k: v for k, v in scores.items() if k not in ['ensemble_cv_mean', 'ensemble_cv_std']}
    models = list(model_scores.keys())
    values = list(model_scores.values())
    
    # バープロット
    bars = plt.bar(models, values)
    
    # アンサンブルモデルのCVスコアをエラーバーで表示
    if 'ensemble_cv_mean' in scores and 'ensemble_cv_std' in scores:
        ensemble_idx = models.index('ensemble')
        plt.errorbar(ensemble_idx, scores['ensemble_cv_mean'],
                    yerr=scores['ensemble_cv_std'],
                    fmt='o', color='red', capsize=5,
                    label='CV Score (mean ± std)')
    
    # グラフの装飾
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # スコアの値をバーの上に表示
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class ModelTrainer:
    def __init__(self, data_path: str, model_save_path: str):
        """
        モデルトレーナーの初期化
        
        Args:
            data_path (str): 学習データのパス
            model_save_path (str): モデル保存先のパス
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.model = None
        self.feature_names = None
        self.label_encoder = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        データの読み込み
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特徴量とラベル
        """
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # 最後の列をラベルとして使用
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # 特徴量名を保存
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Loaded data shape: {X.shape}")
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        モデルの学習
        
        Args:
            X (pd.DataFrame): 特徴量
            y (pd.Series): ラベル
            
        Returns:
            Dict[str, Any]: 学習結果のメトリクス
        """
        # ラベルを数値化
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # アンサンブル用ModelTrainerのインスタンス化
        trainer = EnsembleTrainer(models_dir='models')
        
        # 全てのモデルを学習（最適化あり）
        scores, best_params = trainer.train_all_models(X_train, y_train, X_test, y_test, optimize=True)
        
        # 最良のモデルを取得
        self.model = trainer.best_model
        
        # テストデータでの評価
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 詳細な評価レポート（ラベルを元に戻す）
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'model_scores': scores,
            'best_params': best_params,
            'classification_report': report,
            'feature_names': self.feature_names,
            'label_classes': self.label_encoder.classes_.tolist()
        }
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
        return metrics
    
    def save_model(self) -> None:
        """
        モデルの保存（モデル、特徴量名、ラベルエンコーダーを含む）
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # 保存先ディレクトリの作成
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # モデルとメタデータの保存
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder
        }
        
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model and metadata saved to {self.model_save_path}")

def main():
    # パスの設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    data_path = os.path.join(project_root, 'data', 'train_augmented.csv')
    model_save_path = os.path.join(project_root, 'models', 'model_augmented.pkl')
    
    # モデルトレーナーのインスタンス化
    trainer = ModelTrainer(data_path, model_save_path)
    
    try:
        # データの読み込み
        X, y = trainer.load_data()
        
        # モデルの学習
        metrics = trainer.train_model(X, y)
        
        # 精度の確認
        if metrics['accuracy'] < 0.95:
            logger.warning(
                f"Model accuracy ({metrics['accuracy']:.4f}) is below target (0.95)"
            )
        else:
            logger.info(
                f"Model accuracy ({metrics['accuracy']:.4f}) meets target (0.95)"
            )
        
        # モデルの保存
        trainer.save_model()
        
        # 学習結果の表示
        logger.info("\nBest parameters for each model:")
        for model_name, params in metrics['best_params'].items():
            logger.info(f"\n{model_name}:")
            for param, value in params.items():
                logger.info(f"  {param}: {value}")
        
        logger.info("\nClassification Report:")
        for class_label, scores in metrics['classification_report'].items():
            if isinstance(scores, dict):
                logger.info(f"\nClass {class_label}:")
                for metric, value in scores.items():
                    logger.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
