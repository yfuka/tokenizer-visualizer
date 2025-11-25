# Tokenizer Visualizer

ローカル環境（CPUのみ）で動作する、Python製のトークナイズ可視化Webアプリケーションです。
OpenAIの `tiktoken` (gpt-4o, o1など) や Hugging Faceの `transformers` など、様々なトークナイザーの挙動を視覚的に確認・比較できます。

## 特徴

*   **完全ローカル動作**: データが外部に送信されることはありません。GPUも不要です。
*   **3つのモード**:
    *   **Single Prompt**: テキストを入力してリアルタイムにトークンを確認。複数のトークナイザーを並べて比較可能。
    *   **Chat**: System/User/Assistant ロールごとのメッセージ構成でトークン数を計算。
    *   **JSONL**: データセット（.jsonl）をアップロードし、一括分析および詳細可視化。
        *   `messages` (Chat形式), `prompt`/`response` (Completion形式), `text` (Raw形式) を自動検出。
        *   カラム選択によるフォールバック対応。
*   **高度な可視化**:
    *   **トークンチップ**: トークンごとに色分けされたチップで表示。
    *   **アコーディオン表示**: 1文字が複数のトークンに分割される場合（例: 日本語の一部漢字）、アコーディオンで展開して詳細を確認可能。
    *   **メトリクス切り替え**: 文字数 (Character Count) と単語数 (Word Count) の切り替えが可能。
*   **カスタムトークナイザー**:
    *   Hugging Face Hub からのダウンロード。
    *   ローカルにある `tokenizer.json` 等のアップロード・保存。
*   **モダンなUI**: Streamlitによる使いやすいインターフェース。

## 対応モデル / トークナイザー

*   **Tiktoken**: `gpt-5`, `gpt-4o`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`, `o1` シリーズなど。
*   **Hugging Face**: `bert-base-uncased`, `llama` 系など、`AutoTokenizer` でロード可能なモデル。

## インストールと実行

このプロジェクトは [uv](https://github.com/astral-sh/uv) を使用して管理されています。

### 1. 準備

プロジェクトのディレクトリに移動します。

### 2. 実行

以下のコマンドでアプリケーションを起動します。依存関係は自動的に解決されます。

```bash
uv run streamlit run src/app.py
```

ブラウザが自動的に開き、`http://localhost:8501` でアプリが表示されます。

## 技術スタック

*   Python 3.13+
*   Streamlit
*   Tiktoken
*   Transformers
*   Pandas