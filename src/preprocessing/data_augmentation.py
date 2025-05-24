import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class DataAugmentor:
    def __init__(self, noise_level=0.05, autoencoder_epochs=50, batch_size=32):
        """
        データ拡張クラスの初期化
        
        Args:
            noise_level (float): ガウスノイズの標準偏差
            autoencoder_epochs (int): Autoencoderの学習エポック数
            batch_size (int): バッチサイズ
        """
        self.noise_level = noise_level
        self.autoencoder_epochs = autoencoder_epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.encoder = None
        
    def add_gaussian_noise(self, X: np.ndarray) -> np.ndarray:
        """
        ガウスノイズを追加してデータを拡張
        
        Args:
            X (np.ndarray): 入力データ
            
        Returns:
            np.ndarray: ノイズ付加後のデータ
        """
        noise = np.random.normal(0, self.noise_level, X.shape)
        return X + noise
    
    def build_autoencoder(self, input_dim: int, encoding_dim: int = 3) -> None:
        """
        Autoencoderモデルの構築
        
        Args:
            input_dim (int): 入力次元数
            encoding_dim (int): エンコーディング次元数
        """
        # エンコーダー
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu', name='encoded_layer')(encoded)
        
        # デコーダー
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # モデルの構築
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        # デコーダー部分のみのモデルを作成
        decoder_input = Input(shape=(encoding_dim,))
        x = Dense(16, activation='relu')(decoder_input)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        decoder_output = Dense(input_dim, activation='linear')(x)
        self.decoder = Model(decoder_input, decoder_output)
        # デコーダーの重みをautoencoderからコピー
        self.decoder.layers[1].set_weights(self.autoencoder.layers[5].get_weights())
        self.decoder.layers[2].set_weights(self.autoencoder.layers[6].get_weights())
        self.decoder.layers[3].set_weights(self.autoencoder.layers[7].get_weights())
        
        # モデルのコンパイル
        self.autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                               loss='mse')
    
    def train_autoencoder(self, X: np.ndarray) -> None:
        """
        Autoencoderの学習
        
        Args:
            X (np.ndarray): 学習データ
        """
        if self.autoencoder is None:
            self.build_autoencoder(X.shape[1])
        
        # データのスケーリング
        X_scaled = self.scaler.fit_transform(X)
        
        # モデルの学習
        self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=self.autoencoder_epochs,
            batch_size=self.batch_size,
            verbose=0
        )
    
    def generate_samples(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Autoencoderを使用して新しいサンプルを生成
        
        Args:
            X (np.ndarray): 元のデータ
            n_samples (int): 生成するサンプル数
            
        Returns:
            np.ndarray: 生成されたサンプル
        """
        if self.autoencoder is None:
            input_dim = X.shape[1]
            self.build_autoencoder(input_dim=input_dim, encoding_dim=max(3, input_dim // 2))
            self.train_autoencoder(X)
        
        # データのスケーリング
        X_scaled = self.scaler.transform(X)
        
        # エンコーディング
        encoded = self.encoder.predict(X_scaled)
        
        # ランダムな摂動を加える
        noise = np.random.normal(0, self.noise_level, encoded.shape)
        encoded_noisy = encoded + noise
        
        # デコーダーのみでデコード
        decoded = self.decoder.predict(encoded_noisy)
        
        # スケーリングを元に戻す
        generated_samples = self.scaler.inverse_transform(decoded)
        
        return generated_samples
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                    n_noise_samples: int = 0,
                    n_autoencoder_samples: int = 0) -> tuple:
        """
        データ拡張を実行
        
        Args:
            X (np.ndarray): 特徴量
            y (np.ndarray): ラベル
            n_noise_samples (int): ノイズ付加で生成するサンプル数
            n_autoencoder_samples (int): Autoencoderで生成するサンプル数
            
        Returns:
            tuple: (拡張後の特徴量, 拡張後のラベル)
        """
        X_aug = X.copy()
        y_aug = y.copy()
        
        # ノイズ付加による拡張
        if n_noise_samples > 0:
            # 各クラスから均等にサンプルを選択
            unique_classes = np.unique(y)
            samples_per_class = n_noise_samples // len(unique_classes)
            
            for cls in unique_classes:
                mask = y == cls
                X_cls = X[mask]
                
                # クラス内からランダムにサンプルを選択
                indices = np.random.choice(len(X_cls), samples_per_class, replace=True)
                X_selected = X_cls[indices]
                
                # ノイズを付加
                X_noisy = self.add_gaussian_noise(X_selected)
                
                # 拡張データを追加
                X_aug = np.vstack([X_aug, X_noisy])
                y_aug = np.append(y_aug, np.full(samples_per_class, cls))
        
        # Autoencoderによる拡張
        if n_autoencoder_samples > 0:
            # 各クラスから均等にサンプルを選択
            unique_classes = np.unique(y)
            samples_per_class = n_autoencoder_samples // len(unique_classes)
            
            for cls in unique_classes:
                mask = y == cls
                X_cls = X[mask]
                
                # クラス内からランダムにサンプルを選択
                indices = np.random.choice(len(X_cls), samples_per_class, replace=True)
                X_selected = X_cls[indices]
                
                # Autoencoderで新しいサンプルを生成
                X_generated = self.generate_samples(X_selected, samples_per_class)
                
                # 拡張データを追加
                X_aug = np.vstack([X_aug, X_generated])
                y_aug = np.append(y_aug, np.full(samples_per_class, cls))
        
        return X_aug, y_aug 
