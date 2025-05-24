# pattern

本プロジェクトは、機械学習によるパターン認識APIサーバーです。

## 開発環境セットアップ

DockerとMakefileを利用することで、簡単に開発・運用が可能です。

### 主要コマンド一覧（Makefile）

| コマンド         | 説明                                 |
|------------------|--------------------------------------|
| make build       | Dockerイメージのビルド               |
| make up          | APIサーバーの起動（バックグラウンド） |
| make down        | APIサーバーの停止                     |
| make train       | モデルの再学習                        |
| make logs        | APIサーバーのログ表示                 |
| make test        | テスト実行（pytest）                  |

### 使い方例

1. Dockerイメージのビルド
   ```sh
   make build
   ```
2. APIサーバーの起動
   ```sh
   make up
   ```
3. モデルの再学習
   ```sh
   make train
   ```
4. サーバーの停止
   ```sh
   make down
   ```

---

詳細なAPI仕様は `docs/API.md` を参照してください。
