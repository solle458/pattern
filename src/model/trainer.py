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
        
        os.makedirs(models_dir, exist_ok=True)
        
    def train_random_forest(self, X_train, y_train, X_test, y_test, params=None):
        """Random Forestモデルの学習"""
        if params is None:
            params = {
                'n_estimators': 300,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
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
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        self.models['xgboost'] = {
            'model': model,
            'score': score
        }
        return model, score
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test, params=None):
        """LightGBMモデルの学習"""
        if params is None:
            params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 10,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1e-6,
                'reg_lambda': 1e-6,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        self.models['lightgbm'] = {
            'model': model,
            'score': score
        }
        return model, score
    
    def optimize_hyperparameters(self, X_train, y_train, X_test, y_test, model_name='lightgbm', n_trials=100):
        """Optunaを使用したハイパーパラメータの最適化（クロスバリデーション対応）"""
        def objective(trial):
            if model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    'max_depth': trial.suggest_int('max_depth', 4, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 1e-4, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 1e-4, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 0.1, log=True),
                    'random_state': 42,
                    'verbose': -1
                }
                
                if params['max_depth'] > 8:
                    params['num_leaves'] = min(params['num_leaves'], 2 ** params['max_depth'] - 1)
                
                model = lgb.LGBMClassifier(**params)
                
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
                    'use_label_encoder': False,  # XGBoostの警告を抑制
                    'eval_metric': 'logloss'  # 評価指標を明示的に指定
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
                # クロスバリデーションの代わりに単一のホールドアウトセットで評価
                X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                model.fit(X_train_sub, y_train_sub)
                score = model.score(X_val, y_val)
                return score
            except Exception as e:
                print(f"Warning: Error during optimization: {str(e)}")
                return -1.0
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        try:
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
            # デフォルトのパラメータを返す
            if model_name == 'xgboost':
                return {
                    'n_estimators': 300,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
            elif model_name == 'lightgbm':
                return {
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'min_child_samples': 10,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 1e-6,
                    'reg_lambda': 1e-6,
                    'random_state': 42,
                    'verbose': -1
                }
            else:  # random_forest
                return {
                    'n_estimators': 300,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42,
                    'n_jobs': -1
                }
    
    def create_ensemble(self, X_train, y_train, X_test, y_test, weights=None):
        """アンサンブルモデルの作成と評価"""
        if weights is None:
            weights = [1, 1, 1]
        
        estimators = [
            ('rf', self.models['random_forest']['model']),
            ('xgb', self.models['xgboost']['model']),
            ('lgb', self.models['lightgbm']['model'])
        ]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        ensemble.fit(X_train, y_train)
        
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=self.cv, scoring='accuracy')
        test_score = ensemble.score(X_test, y_test)
        
        self.models['ensemble'] = {
            'model': ensemble,
            'score': test_score,
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std()
        }
        
        return ensemble, test_score, cv_scores

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
                train_scores = cv_results[f'train_{metric}']
                test_scores = cv_results[f'test_{metric}']
                
                results[metric] = {
                    'train_mean': np.mean(train_scores),
                    'train_std': np.std(train_scores),
                    'test_mean': np.mean(test_scores),
                    'test_std': np.std(test_scores),
                    'scores': test_scores.tolist()
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
                           test_mean - test_std,
                           test_mean + test_std,
                           alpha=0.1)
            
            plt.axhline(y=train_mean, color='r', linestyle='--', label='Train Mean')
            plt.fill_between(x,
                           train_mean - train_std,
                           train_mean + train_std,
                           alpha=0.1)
            
            plt.title(f'{metric.capitalize()} Learning Curve')
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
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_test, y_test, cv=self.cv, scoring='accuracy')
        
        return {
            'accuracy': accuracy,
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        } 
