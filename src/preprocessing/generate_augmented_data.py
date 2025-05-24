import os
import pandas as pd
import numpy as np
from src.preprocessing.preprocessor import DataPreprocessor
import logging

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_augmented_data(
    input_path='data/train.csv',
    output_path='data/train_augmented.csv',
    n_noise_samples=500,  # ノイズ付加で生成するサンプル数
    n_autoencoder_samples=500,  # Autoencoderで生成するサンプル数
    noise_level=0.05,
    autoencoder_epochs=50
):
    """
    データ拡張を実行し、拡張データを保存する
    
    Args:
        input_path (str): 入力データのパス
        output_path (str): 出力データのパス
        n_noise_samples (int): ノイズ付加で生成するサンプル数
        n_autoencoder_samples (int): Autoencoderで生成するサンプル数
        noise_level (float): ガウスノイズの標準偏差
        autoencoder_epochs (int): Autoencoderの学習エポック数
    """
    try:
        # データプリプロセッサの初期化
        preprocessor = DataPreprocessor(
            data_path=input_path,
            use_data_augmentation=True,
            noise_level=noise_level,
            autoencoder_epochs=autoencoder_epochs
        )
        
        # データの読み込みと拡張
        logger.info("データの読み込みと拡張を開始します...")
        df = preprocessor.load_data()
        X = df[preprocessor.feature_columns].values
        y = df[preprocessor.target_column].values
        
        # データの形状をログ出力
        logger.info(f"入力データの形状: X={X.shape}, y={y.shape}")
        logger.info(f"特徴量の列名: {preprocessor.feature_columns}")
        
        # ラベルの数値変換
        y_encoded = preprocessor.label_encoder.fit_transform(y)
        
        # データ拡張の実行
        X_aug, y_aug = preprocessor.data_augmentor.augment_data(
            X, y_encoded,
            n_noise_samples=n_noise_samples,
            n_autoencoder_samples=n_autoencoder_samples
        )
        
        # ラベルを元の形式に戻す
        y_aug_labels = preprocessor.label_encoder.inverse_transform(y_aug)
        
        # 拡張データをDataFrameに変換
        df_aug = pd.DataFrame(X_aug, columns=[f'param{i+1}' for i in range(X_aug.shape[1])])
        df_aug['class'] = y_aug_labels
        
        # 出力ディレクトリの作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 拡張データの保存
        df_aug.to_csv(output_path, index=False)
        
        # 結果の表示
        original_samples = len(X)
        augmented_samples = len(X_aug)
        logger.info(f"元のデータ数: {original_samples}")
        logger.info(f"拡張後のデータ数: {augmented_samples}")
        logger.info(f"生成されたサンプル数: {augmented_samples - original_samples}")
        
        # クラスごとのサンプル数を表示
        class_counts = df_aug['class'].value_counts()
        logger.info("\nクラスごとのサンプル数:")
        for cls, count in class_counts.items():
            logger.info(f"クラス {cls}: {count}サンプル")
        
        logger.info(f"\n拡張データを {output_path} に保存しました。")
        
    except Exception as e:
        logger.error(f"データ拡張中にエラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    # デフォルトパラメータでデータ拡張を実行
    generate_augmented_data() 
