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

# APIサーバーのログ表示
logs:
	docker compose logs -f api

# テスト実行（pytest）
test:
	docker compose run --rm api pytest 
