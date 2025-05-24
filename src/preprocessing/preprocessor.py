import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler
import joblib
import os
from src.preprocessing.data_augmentation import DataAugmentor

class DataPreprocessor:
    def __init__(self, data_path='data/train.csv', add_poly_features=True, add_interaction_only=True,
                 use_data_augmentation=False, noise_level=0.05, autoencoder_epochs=50):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [0, 1, 2, 3, 4, 5]  # 特徴量の列インデックス（6次元）
        self.target_column = 6  # ラベルの列インデックス
        self.add_poly_features = add_poly_features
        self.add_interaction_only = add_interaction_only
        self.poly = None
        self.use_data_augmentation = use_data_augmentation
        if use_data_augmentation:
            self.data_augmentor = DataAugmentor(
                noise_level=noise_level,
                autoencoder_epochs=autoencoder_epochs
            )
        
    def load_data(self):
        """データの読み込み"""
        df = pd.read_csv(self.data_path, header=None)
        return df
        
    def prepare_data(self, test_size=0.2, random_state=42,
                    n_noise_samples=0, n_autoencoder_samples=0):
        """データの前処理と分割"""
        # データ読み込み
        df = self.load_data()
        
        # 特徴量とラベルの分離
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        # ラベルの数値変換
        y = self.label_encoder.fit_transform(y)
        
        # データ拡張（オプション）
        if self.use_data_augmentation and (n_noise_samples > 0 or n_autoencoder_samples > 0):
            X, y = self.data_augmentor.augment_data(
                X, y,
                n_noise_samples=n_noise_samples,
                n_autoencoder_samples=n_autoencoder_samples
            )
        
        # 特徴量エンジニアリング
        X = self._engineer_features(X)
        
        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # スケーリング（RobustScalerを使用）
        self.scaler = RobustScaler()  # 外れ値に強いスケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def _engineer_features(self, X):
        """特徴量エンジニアリング"""
        # 基本的な統計量の計算
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True)
        X_max = np.max(X, axis=1, keepdims=True)
        X_min = np.min(X, axis=1, keepdims=True)
        
        # 非線形変換（値の範囲を考慮）
        X_scaled = (X - X_mean) / (X_std + 1e-8)  # 標準化
        X_nonlinear = np.column_stack([
            X_scaled,  # 標準化された特徴量
            np.sin(X_scaled),  # 周期関数
            np.cos(X_scaled),
            np.tanh(X_scaled),  # シグモイド関数
            np.abs(X_scaled),  # 絶対値
            np.square(X_scaled),  # 2乗
            X_mean,  # 統計量
            X_std,
            X_max,
            X_min,
            (X_max - X_min) / (X_std + 1e-8),  # 変動係数
        ])
        
        # 多項式特徴量の追加
        if self.add_poly_features:
            self.poly = PolynomialFeatures(
                degree=2,
                interaction_only=self.add_interaction_only,
                include_bias=False
            )
            X_poly = self.poly.fit_transform(X_scaled)  # 標準化された特徴量に対して多項式特徴量を生成
            
            # 元の特徴量と非線形変換、多項式特徴量を結合
            X = np.column_stack([X_nonlinear, X_poly])
        else:
            X = X_nonlinear
        
        # 欠損値の処理
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X
    
    def save_preprocessors(self, scaler_path='models/scaler.joblib', label_encoder_path='models/label_encoder.joblib', poly_path='models/poly.joblib'):
        """前処理器を保存"""
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        if self.poly is not None:
            joblib.dump(self.poly, poly_path)
    
    def load_preprocessors(self, scaler_path='models/scaler.joblib', label_encoder_path='models/label_encoder.joblib', poly_path='models/poly.joblib'):
        """前処理器を読み込み"""
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)
        if os.path.exists(poly_path):
            self.poly = joblib.load(poly_path)
    
    def transform(self, X):
        """新しいデータの変換"""
        if self.poly is not None:
            X = self.poly.transform(X)
        return self.scaler.transform(X)
    
    def inverse_transform_labels(self, y):
        """数値ラベルを元の文字列ラベルに戻す"""
        return self.label_encoder.inverse_transform(y) 
