from flask import Flask
from flask.json import jsonify
import logging
from pathlib import Path
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Flaskアプリケーションのファクトリー関数"""
    app = Flask(__name__)

    # モデルの読み込み
    model_name = os.getenv('MODEL_NAME', 'model_augmented.pkl')  # デフォルト値を設定
    model_path = Path(__file__).parent.parent.parent / 'models' / model_name
    logger.info(f"Loading model from: {model_path}")
    try:
        loaded_data = joblib.load(model_path)
        # モデルが辞書形式で保存されている場合
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            app.model = loaded_data['model']
            logger.info(f"Model loaded successfully from dictionary (type: {type(app.model)})")
        else:
            app.model = loaded_data
            logger.info(f"Model loaded successfully (type: {type(app.model)})")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        app.model = None

    # ルートの登録
    from .routes import register_routes
    register_routes(app)

    # エラーハンドラーの登録
    from .errors import register_error_handlers
    register_error_handlers(app)

    return app

def run_app():
    """アプリケーションの実行"""
    app = create_app()
    return app

if __name__ == '__main__':
    app = run_app()
    app.run(host='0.0.0.0', port=5001, debug=True) 
