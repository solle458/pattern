import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
    make_scorer
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split,
    cross_validate, KFold
)
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import os
from datetime import datetime
import json
from typing import Dict, Any, List, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import NotFittedError
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, models_dir='models', n_splits=5, cv_strategy='stratified'):
        """
        モデルトレーナーの初期化
        
        Args:
            models_dir (str): モデル保存ディレクトリ
            n_splits (int): 交差検証の分割数
            cv_strategy (str): 交差検証の戦略 ('stratified' or 'kfold')
        """
        self.models_dir = models_dir
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.n_splits = n_splits
        self.cv_strategy = cv_strategy
        
        # 交差検証の設定
        if cv_strategy == 'stratified':
            self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 評価指標の設定
        self.scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc_ovr'  # 多クラス分類用
        }
        # 多クラス分類の場合、accuracy以外の指標はweightedまたはovrを使用していることを明記
        self._scoring_titles = {
            'accuracy': 'Accuracy Learning Curve',
            'precision': 'Precision (Weighted) Learning Curve',
            'recall': 'Recall (Weighted) Learning Curve',
            'f1': 'F1 (Weighted) Learning Curve',
            'roc_auc': 'Roc_auc (OvR) Learning Curve'
        }
        
        os.makedirs(models_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        
    def train_random_forest(self, X_train, y_train, X_test, y_test, params=None):
        """Random Forestモデルの学習"""
        if params is None:
            params = {
                'n_estimators': 200,  # データサイズに応じて調整
                'max_depth': 4,       # 特徴量数が少ないため浅めに
                'min_samples_split': 5,  # 過学習防止
                'min_samples_leaf': 3,   # 過学習防止
                'max_features': 'sqrt',  # 特徴量数が少ないため
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'  # クラス不均衡に対応
            }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        self.models['random_forest'] = {
            'model': model,
            'score': score
        }
        return model, score
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, params=None):
        """XGBoostモデルの学習"""
        if params is None:
            params = {
                'n_estimators': 200,     # データサイズに応じて調整
                'max_depth': 3,          # 浅めの木構造
                'learning_rate': 0.05,   # 適度な学習率
                'min_child_weight': 3,   # 過学習防止
                'subsample': 0.8,        # バギング
                'colsample_bytree': 0.8, # 特徴量のサブサンプリング
                'gamma': 0.1,            # 正則化
                'reg_alpha': 0.5,        # L1正則化
                'reg_lambda': 0.5,       # L2正則化
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'mlogloss',
                'objective': 'multi:softprob',
                'tree_method': 'hist'    # メモリ効率の良い方法
            }
        model = xgb.XGBClassifier(**params)
        
        # 早期停止のために学習セットを分割
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # 早期停止を伴う学習（新しいAPIを使用）
        try:
            model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    xgb.callback.EarlyStopping(
                        rounds=20,
                        save_best=True,
                        maximize=False
                    )
                ]
            )
        except Exception as e:
            print(f"Warning: Error during XGBoost training with early stopping: {str(e)}")
            # 早期停止なしで学習を実行
            model.fit(X_train, y_train)

        # テストデータでの評価
        score = model.score(X_test, y_test)
        self.models['xgboost'] = {
            'model': model,
            'score': score
        }
        return model, score
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test, params=None):
        """LightGBMモデルの学習"""
        # 特徴量のスケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        if params is None:
            # クラスの分布を計算
            class_counts = np.bincount(y_train)
            class_weights = len(y_train) / (len(np.unique(y_train)) * class_counts)
            class_c3_idx = np.where(np.unique(y_train) == 2)[0][0]
            class_weights[class_c3_idx] *= 1.2
            
            params = {
                'n_estimators': 200,      # データサイズに応じて調整
                'max_depth': 3,           # 浅めの木構造
                'learning_rate': 0.05,    # 適度な学習率
                'num_leaves': 8,          # 2^max_depth以下に制限
                'min_child_samples': 5,   # 過学習防止
                'subsample': 0.8,         # バギング
                'colsample_bytree': 0.8,  # 特徴量のサブサンプリング
                'reg_alpha': 0.5,         # L1正則化
                'reg_lambda': 0.5,        # L2正則化
                'min_child_weight': 3,    # 過学習防止
                'min_split_gain': 0.1,    # 分割の閾値
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'class_weight': dict(enumerate(class_weights)),
                'is_unbalance': True,
                'boost_from_average': True,
                'bagging_freq': 5,        # バギングの頻度
                'bagging_fraction': 0.8,  # バギングの割合
                'feature_fraction': 0.8,  # 特徴量のサブサンプリング
                'drop_rate': 0.1,         # ドロップアウト
                'max_bin': 63            # メモリ効率のため制限
            }
        
        model = lgb.LGBMClassifier(**params)
        
        # 早期停止のために学習セットを分割
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # 早期停止を伴う学習
        try:
            model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
        except Exception as e:
            print(f"Warning: Error during LightGBM training with early stopping: {str(e)}")
            model.fit(X_train_scaled, y_train)
        
        # スケーリングされたテストデータで評価
        score = model.score(X_test_scaled, y_test)
        self.models['lightgbm'] = {
            'model': model,
            'score': score,
            'scaler': self.scaler  # スケーラーを保存
        }
        
        # 学習後に特徴量重要度を分析
        feature_importance = self.analyze_feature_importance(model, X_train_scaled, y_train, class_label=2)
        
        # 重要度の高い特徴量をログ出力
        logger.info("\nTop 10 important features for class C3:")
        for feature, importance in list(feature_importance.items())[:10]:
            logger.info(f"{feature}: {importance:.4f}")
        
        # 重要度の低い特徴量もログ出力
        logger.info("\nBottom 10 important features for class C3:")
        for feature, importance in list(feature_importance.items())[-10:]:
            logger.info(f"{feature}: {importance:.4f}")
        
        # C3以外のクラス性能も詳細にログ出力
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        for class_label in [0, 1, 3, 4]:
            logger.info(f"\nClass C{class_label+1} performance for lightgbm:")
            logger.info(f"precision: {report[str(class_label)]['precision']:.4f}")
            logger.info(f"recall: {report[str(class_label)]['recall']:.4f}")
            logger.info(f"f1-score: {report[str(class_label)]['f1-score']:.4f}")
            logger.info(f"support: {report[str(class_label)]['support']:.0f}")
        
        return model, score
    
    def optimize_hyperparameters(self, X_train, y_train, X_test, y_test, model_name='lightgbm', n_trials=100):
        """Optunaを使用したハイパーパラメータの最適化（クロスバリデーション対応）"""
        def objective(trial):
            if model_name == 'lightgbm':
                # 特徴量のスケーリング
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                
                # クラスの分布を計算
                class_counts = np.bincount(y_train)
                class_weights = len(y_train) / (len(np.unique(y_train)) * class_counts)
                class_c3_idx = np.where(np.unique(y_train) == 2)[0][0]
                class_weights[class_c3_idx] *= 1.2
                
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),  # 範囲を制限
                    'max_depth': trial.suggest_int('max_depth', 2, 4),           # より浅い木構造
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 4, 16),        # 2^max_depthに基づいて制限
                    'min_child_samples': trial.suggest_int('min_child_samples', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 2, 5),
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.05, 0.2, log=True),
                    'random_state': 42,
                    'verbose': -1,
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'class_weight': dict(enumerate(class_weights)),
                    'is_unbalance': True,
                    'boost_from_average': True,
                    'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),
                    'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.2),
                    'max_bin': 63
                }
                
                # max_depthとnum_leavesの関係を制御
                params['num_leaves'] = min(params['num_leaves'], 2 ** params['max_depth'])
                
                model = lgb.LGBMClassifier(**params)
                
                try:
                    cv_results = cross_validate(
                        model, X_train_scaled, y_train,
                        cv=self.cv,
                        scoring=self.scoring,
                        return_train_score=False,
                        n_jobs=-1
                    )
                    
                    # クラスC3のF1スコアも考慮
                    mean_accuracy = np.mean(cv_results['test_accuracy'])
                    return mean_accuracy
                    
                except Exception as e:
                    print(f"Warning: Error during optimization trial: {str(e)}")
                    return -np.inf
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42,
                    'eval_metric': 'logloss', # 多クラス分類の場合に適宜変更
                    'objective': 'multi:softprob' # 多クラス分類の場合に適宜変更
                }
                model = xgb.XGBClassifier(**params)
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            try:
                # クロスバリデーションを使用して評価
                # Optunaの目的関数は単一のfloatを返す必要があるため、平均テストスコアを使用
                cv_results = cross_validate(
                    model, X_train, y_train,
                    cv=self.cv,
                    scoring=self.scoring, # 定義済みの複数の指標で評価
                    return_train_score=False, # チューニング時はテストスコアのみで十分
                    n_jobs=-1
                )
                
                # 最適化の指標として平均Accuracyを使用
                mean_accuracy = np.mean(cv_results['test_accuracy'])
                return mean_accuracy
                
            except Exception as e:
                print(f"Warning: Error during optimization trial: {str(e)}")
                return -np.inf # エラーの場合は非常に低い値を返してペナルティを与える
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        try:
            # Optunaの学習でプログレスバーを表示
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=1200,
                show_progress_bar=True
            )
            
            print(f"\nBest trial for {model_name}:")
            print(f"  Value: {study.best_trial.value:.4f}")
            print("  Params: ")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
            
            return study.best_params
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            # エラー発生時もデフォルトのパラメータを返す
            if model_name == 'xgboost':
                return {
                    'n_estimators': 300,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'objective': 'multi:softprob'
                }
            elif model_name == 'lightgbm':
                return {
                    'n_estimators': 500,
                    'max_depth': 3,
                    'learning_rate': 0.05,
                    'num_leaves': 15,       # 2^max_depth以下
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 1,
                    'reg_lambda': 1,
                    'random_state': 42,
                    'drop_rate': 0.1
                }
            else:  # random_forest
                return {
                    'n_estimators': 300,
                    'max_depth': 15, # デフォルトを少し制限
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42,
                    'n_jobs': -1
                }
    
    def create_ensemble(self, X_train, y_train, X_test, y_test, weights=None):
        """アンサンブルモデルの作成と評価（メモリ最適化版）"""
        if weights is None:
            weights = [1, 1, 1]
        
        try:
            # 各モデルの予測を逐次的に計算
            predictions = []
            valid_models = []
            
            for name, model_info in self.models.items():
                if name != 'ensemble':  # 既存のアンサンブルモデルは除外
                    try:
                        model = model_info['model']
                        if model is not None:  # モデルがNoneでないことを確認
                            valid_models.append((name, model))
                            # 予測確率を計算
                            if hasattr(model, 'predict_proba'):
                                pred = model.predict_proba(X_test)
                            else:
                                # predict_probaが利用できない場合は、one-hotエンコーディングで対応
                                pred = np.zeros((len(X_test), len(np.unique(y_train))))
                                pred[np.arange(len(X_test)), model.predict(X_test)] = 1
                            predictions.append(pred)
                    except Exception as e:
                        logger.error(f"Error in {name} model prediction: {str(e)}")
                        continue
            
            if not predictions:
                raise ValueError("No valid predictions available for ensemble")
            
            # 予測の重み付け平均を計算
            weighted_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_pred += weights[i] * pred
            weighted_pred /= sum(weights)
            
            # 最終的な予測を取得
            y_pred = np.argmax(weighted_pred, axis=1)
            
            # アンサンブルモデルの評価
            test_score = accuracy_score(y_test, y_pred)
            
            # VotingClassifierを使用してアンサンブルモデルを作成
            estimators = [(name, model) for name, model in valid_models]
            ensemble_model = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights,
                n_jobs=-1
            )
            
            # アンサンブルモデルの学習
            ensemble_model.fit(X_train, y_train)
            
            # アンサンブルモデルの保存
            self.models['ensemble'] = {
                'model': ensemble_model,
                'score': test_score,
                'weights': weights,
                'predictions': weighted_pred
            }
            
            # 交差検証の実行（メモリ効率を考慮）
            cv_scores = []
            for train_idx, val_idx in self.cv.split(X_train, y_train):
                X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                
                # アンサンブルモデルの学習と評価
                fold_model = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=weights,
                    n_jobs=-1
                )
                fold_model.fit(X_train_fold, y_train_fold)
                fold_pred = fold_model.predict(X_val_fold)
                cv_scores.append(accuracy_score(y_val_fold, fold_pred))
            
            cv_scores = np.array(cv_scores)
            
            return ensemble_model, test_score, cv_scores
            
        except Exception as e:
            logger.error(f"Error in ensemble creation: {str(e)}")
            # エラー発生時はデフォルトのVotingClassifierを返す
            try:
                default_ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in self.models.items() if name != 'ensemble' and model is not None],
                    voting='soft',
                    n_jobs=-1
                )
                default_ensemble.fit(X_train, y_train)
                return default_ensemble, 0.0, np.array([0.0])
            except Exception as e2:
                logger.error(f"Error creating default ensemble: {str(e2)}")
                return None, 0.0, np.array([0.0])

    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        モデルの交差検証を実行
        
        Args:
            model: 学習済みモデル
            X (pd.DataFrame): 特徴量
            y (pd.Series): ラベル
            
        Returns:
            Dict[str, Any]: 交差検証の結果
        """
        try:
            # 交差検証の実行
            cv_results = cross_validate(
                model, X, y,
                cv=self.cv,
                scoring=self.scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # 結果の集計
            results = {}
            for metric in self.scoring.keys():
                # 存在しないスコアキーに対するエラーハンドリングを追加
                train_key = f'train_{metric}'
                test_key = f'test_{metric}'
                
                if train_key in cv_results and test_key in cv_results:
                    train_scores = cv_results[train_key]
                    test_scores = cv_results[test_key]
                    
                    results[metric] = {
                        'train_mean': np.mean(train_scores),
                        'train_std': np.std(train_scores),
                        'test_mean': np.mean(test_scores),
                        'test_std': np.std(test_scores),
                        'scores': test_scores.tolist()
                    }
                else:
                    print(f"Warning: Scores not found for metric '{metric}' in cross-validation results.")
                    # 存在しない場合は空の辞書またはエラーを示す値を設定
                    results[metric] = {
                        'train_mean': np.nan, 'train_std': np.nan,
                        'test_mean': np.nan, 'test_std': np.nan,
                        'scores': []
                    }
            
            # 学習曲線のプロット
            self._plot_learning_curves(model, X, y, results)
            
            return results
            
        except Exception as e:
            print(f"Error during cross-validation: {str(e)}")
            return None

    def _plot_learning_curves(self, model, X: pd.DataFrame, y: pd.Series, cv_results: Dict[str, Any]):
        """
        学習曲線のプロット
        
        Args:
            model: 学習済みモデル
            X (pd.DataFrame): 特徴量
            y (pd.Series): ラベル
            cv_results (Dict[str, Any]): 交差検証の結果
        """
        plt.figure(figsize=(15, 10))
        
        # 各評価指標の学習曲線をプロット
        for i, (metric, scores) in enumerate(cv_results.items(), 1):
            plt.subplot(2, 3, i)
            
            train_mean = scores['train_mean']
            train_std = scores['train_std']
            test_mean = scores['test_mean']
            test_std = scores['test_std']
            
            x = np.arange(1, self.n_splits + 1)
            
            plt.plot(x, scores['scores'], 'o-', label='Test Score')
            plt.fill_between(x,
                           np.maximum(0, test_mean - test_std), # スコアが0以下にならないようにクリップ
                           np.minimum(1, test_mean + test_std), # スコアが1以上にならないようにクリップ
                           alpha=0.1)
            
            plt.axhline(y=train_mean, color='r', linestyle='--', label='Train Mean')
            plt.fill_between(x,
                           np.maximum(0, train_mean - train_std), # スコアが0以下にならないようにクリップ
                           np.minimum(1, train_mean + train_std), # スコアが1以上にならないようにクリップ
                           alpha=0.1)
            
            # 更新: 評価指標の種類を反映したタイトルを使用
            plt.title(self._scoring_titles.get(metric, f'{metric.capitalize()} Learning Curve'))
            plt.xlabel('Fold')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # 学習曲線の保存
        save_path = os.path.join(self.models_dir, 'learning_curves.png')
        plt.savefig(save_path)
        plt.close()

    def train_all_models(self, X_train, y_train, X_test, y_test, optimize=False):
        """全てのモデルを学習し、アンサンブルも作成（交差検証対応）"""
        best_params = {}
        cv_results = {}
        
        # 各モデルの学習と交差検証
        for model_name, train_func in [
            ('random_forest', self.train_random_forest),
            ('xgboost', self.train_xgboost),
            ('lightgbm', self.train_lightgbm)
        ]:
            print(f"\nTraining {model_name}...")
            
            if optimize:
                print(f"Optimizing {model_name}...")
                model_params = self.optimize_hyperparameters(
                    X_train, y_train, X_test, y_test,
                    model_name=model_name
                )
                best_params[model_name] = model_params
                model, _ = train_func(X_train, y_train, X_test, y_test, model_params)
            else:
                model, _ = train_func(X_train, y_train, X_test, y_test)
            
            # 交差検証の実行
            print(f"Performing cross-validation for {model_name}...")
            cv_result = self.cross_validate_model(model, X_train, y_train)
            cv_results[model_name] = cv_result
            
            # モデルの保存
            self.models[model_name] = {
                'model': model,
                'cv_results': cv_result
            }
        
        # アンサンブルモデルの作成
        print("\nCreating ensemble model...")
        ensemble, ensemble_score, _ = self.create_ensemble(X_train, y_train, X_test, y_test)
        
        # アンサンブルモデルの交差検証
        ensemble_cv_result = self.cross_validate_model(ensemble, X_train, y_train)
        cv_results['ensemble'] = ensemble_cv_result
        
        self.models['ensemble'] = {
            'model': ensemble,
            'score': ensemble_score,
            'cv_results': ensemble_cv_result
        }
        
        # 最良モデルの選択（交差検証の結果に基づく）
        best_model_name = max(
            cv_results.keys(),
            key=lambda x: cv_results[x]['accuracy']['test_mean']
        )
        self.best_model = self.models[best_model_name]['model']
        self.best_score = cv_results[best_model_name]['accuracy']['test_mean']
        
        # クラスC3の性能を重点的に評価
        for model_name, model_info in self.models.items():
            if model_name != 'ensemble':  # 個別モデルのみ評価
                model = model_info['model']
                y_pred = model.predict(X_test)
                
                # クラスC3の性能指標を計算
                c3_mask = (y_test == 2)  # C3のラベルは2と仮定
                c3_accuracy = accuracy_score(y_test[c3_mask], y_pred[c3_mask])
                c3_report = classification_report(
                    y_test[c3_mask], y_pred[c3_mask],
                    output_dict=True
                )
                
                logger.info(f"\nClass C3 performance for {model_name}:")
                logger.info(f"Accuracy: {c3_accuracy:.4f}")
                logger.info("Classification Report:")
                for metric, value in c3_report.items():
                    if isinstance(value, dict):
                        logger.info(f"{metric}: {value['f1-score']:.4f}")
        
        return cv_results, best_params
    
    def save_models(self):
        """モデルとメタデータの保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_info = {
            'timestamp': timestamp,
            'best_model': None,
            'best_score': self.best_score,
            'model_scores': {name: info['score'] for name, info in self.models.items()}
        }
        
        for name, info in self.models.items():
            model_path = os.path.join(self.models_dir, f'{name}_{timestamp}.joblib')
            joblib.dump(info['model'], model_path)
            if info['model'] == self.best_model:
                model_info['best_model'] = name
                best_model_path = os.path.join(self.models_dir, 'best_model.joblib')
                joblib.dump(info['model'], best_model_path)
        
        meta_path = os.path.join(self.models_dir, f'model_info_{timestamp}.json')
        with open(meta_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        return model_info
    
    def evaluate_model(self, model, X_test, y_test):
        """モデルの評価（クロスバリデーション対応）"""
        # モデルが学習済みかチェック
        try:
            y_pred = model.predict(X_test)
        except NotFittedError:
            print("Error: Model is not fitted yet.")
            return None
            
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # テストセットに対してクロスバリデーションを行うのは一般的ではないため削除
        # cv_scores = cross_val_score(model, X_test, y_test, cv=self.cv, scoring='accuracy')
        
        return {
            'accuracy': accuracy,
            # 'cv_score_mean': cv_scores.mean(), # 削除
            # 'cv_score_std': cv_scores.std(),   # 削除
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }

    def analyze_feature_importance(self, model, X: pd.DataFrame, y: pd.Series, class_label: int = None) -> Dict[str, float]:
        """
        特徴量重要度の分析
        
        Args:
            model: 学習済みモデル
            X (pd.DataFrame): 特徴量
            y (pd.Series): ラベル
            class_label (int, optional): 特定のクラスのラベル
            
        Returns:
            Dict[str, float]: 特徴量名と重要度の辞書
        """
        if class_label is not None:
            # 特定のクラスに焦点を当てた分析
            mask = (y == class_label)
            X_subset = X[mask]
            y_subset = y[mask]
        else:
            X_subset = X
            y_subset = y
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_booster'):
            importances = model.get_booster().get_score(importance_type='weight')
            importances = [importances.get(f'f{i}', 0) for i in range(X.shape[1])]
        else:
            return {}
        
        # 特徴量名と重要度を辞書に変換
        feature_importance = dict(zip(X.columns, importances))
        
        # 重要度でソート
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance 
