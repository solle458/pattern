# 開発用Makefile

# Dockerイメージのビルド
build:
	docker compose build --no-cache

# APIサーバーの起動（バックグラウンド）
up:
	docker compose up -d

# APIサーバーの停止
down:
	docker compose down

# モデルの再学習
train:
	docker compose run --rm api python src/model/train.py

# データの分析
analyze:
	docker compose run --rm api python src/preprocess/data_analysis.py

# APIサーバーのログ表示
logs:
	docker compose logs -f api

# テスト実行（pytest）
test:
	docker compose run --rm api pytest 

# データ拡張の実行
augment-data:
	docker compose run --rm api bash -c "PYTHONPATH=.:$$PYTHONPATH python src/preprocessing/generate_augmented_data.py"
