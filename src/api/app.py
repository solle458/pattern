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
    model_name = os.getenv('MODEL_NAME', 'model_augmented.pkl')
    # モデルのパスを環境変数から取得、なければデフォルトパスを使用
    base_model_path = os.getenv('MODEL_BASE_PATH', str(Path(__file__).parent.parent.parent))
    model_path = Path(base_model_path) / 'models' / model_name
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

# グローバルなappインスタンスを作成
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False) 
