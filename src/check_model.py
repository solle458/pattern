import pickle
import joblib
import logging
from pathlib import Path

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_model(model_path):
    """モデルファイルの内容を確認する"""
    logger.info(f"Checking model file: {model_path}")
    
    try:
        # まずpickleで読み込みを試みる
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            logger.info("Successfully loaded with pickle")
    except Exception as e:
        logger.error(f"Failed to load with pickle: {str(e)}")
        try:
            # pickleで失敗した場合、joblibで試みる
            model_data = joblib.load(model_path)
            logger.info("Successfully loaded with joblib")
        except Exception as e:
            logger.error(f"Failed to load with joblib: {str(e)}")
            return
    
    # モデルデータの型を確認
    logger.info(f"Model data type: {type(model_data)}")
    
    # 辞書の場合、キーを確認
    if isinstance(model_data, dict):
        logger.info("Model data is a dictionary")
        logger.info(f"Available keys: {list(model_data.keys())}")
        
        # modelキーがある場合、その内容を確認
        if 'model' in model_data:
            model = model_data['model']
            logger.info(f"Model object type: {type(model)}")
            
            # モデルオブジェクトのメソッドを確認
            methods = [method for method in dir(model) if not method.startswith('_')]
            logger.info(f"Available methods: {methods}")
            
            # 必要なメソッドの存在確認
            required_methods = ['predict', 'predict_proba']
            for method in required_methods:
                if hasattr(model, method):
                    logger.info(f"Method '{method}' is available")
                else:
                    logger.error(f"Method '{method}' is NOT available")
    else:
        # 辞書でない場合、直接モデルオブジェクトとして扱う
        model = model_data
        logger.info(f"Model object type: {type(model)}")
        
        # モデルオブジェクトのメソッドを確認
        methods = [method for method in dir(model) if not method.startswith('_')]
        logger.info(f"Available methods: {methods}")
        
        # 必要なメソッドの存在確認
        required_methods = ['predict', 'predict_proba']
        for method in required_methods:
            if hasattr(model, method):
                logger.info(f"Method '{method}' is available")
            else:
                logger.error(f"Method '{method}' is NOT available")

if __name__ == '__main__':
    model_path = Path('/app/models/model_augmented.pkl')
    check_model(model_path) 
