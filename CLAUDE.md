# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Moonshine Voice — リアルタイム音声アプリケーション向けのオンデバイス音声認識(STT)ライブラリ。C++20コアに Python / Swift / Android(JNI) のバインディングを持つクロスプラットフォーム構成。8言語以上対応（英語はMITライセンス、他言語はCommunity License）。

## Python環境構築（uv）

### 初期セットアップ

```bash
uv python pin 3.12
uv add moonshine-voice
```

現在の `pyproject.toml` の依存関係:

- `moonshine-voice>=0.0.49` — STTライブラリ本体（PyPIから。ネイティブ共有ライブラリ同梱）
- `datasets>=4.6.1` — HuggingFaceデータセット読み込み用
- `soundfile>=0.13.1` — 音声ファイルの読み書き

依存パッケージの追加は `uv add <package>` で行う。`uv.lock` は `.gitignore` 済み。

### 仮想環境の復元

```bash
uv sync
```

## 推論の実行

### WAVファイルの書き起こし（英語）

```bash
uv run python examples/python/basic_transcription.py
```

内蔵テスト音声 `two_cities.wav` を使い、非ストリーミングとストリーミング両方で書き起こしを実行する。初回実行時にモデル（約290MB）が自動ダウンロードされる。

任意のWAVファイルを指定する場合:

```bash
uv run python examples/python/basic_transcription.py path/to/audio.wav
```

### 日本語モデルでの書き起こし

```bash
uv run python examples/python/basic_transcription.py --language ja path/to/japanese_audio.wav
```

対応言語コード: `en`（英語）, `ja`（日本語）, `es`（スペイン語）, `ko`（韓国語）, `zh`（中国語）, `ar`（アラビア語）, `uk`（ウクライナ語）, `vi`（ベトナム語）。言語ごとに専用モデルが初回実行時に自動ダウンロードされる。

### マイク入力でリアルタイム認識

```bash
uv run python examples/python/mic_transcription.py
uv run python examples/python/mic_transcription.py --language ja
```

### 音声コマンド認識（Intent Recognition）

```bash
uv run python examples/python/intent_recognition.py
```

### テスト用日本語音声の取得

`test-assets/ja/` に日本語テスト音声を配置する（`.gitignore` 済み）。ReazonSpeechテストセットからダウンロードする例:

```python
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import pyarrow as pa
import soundfile as sf
import io

parquet_path = hf_hub_download(
    'japanese-asr/ja_asr.reazonspeech_test',
    'data/test-00000-of-00002.parquet',
    repo_type='dataset'
)
pf = pq.ParquetFile(parquet_path)
batch = next(pf.iter_batches(batch_size=3))
table = pa.Table.from_batches([batch])

for i in range(3):
    audio = table.column('audio')[i].as_py()
    data, sr = sf.read(io.BytesIO(audio['bytes']))
    sf.write(f'test-assets/ja/reazonspeech_{i:03d}.wav', data, sr)
```

## C++コアのビルドとテスト

### Windows

```bat
scripts\run-core-tests.bat
```

内部で `core/build` にCMakeビルドを行い、全テストを実行する。手動で行う場合:

```bat
cd core\build
cmake ..
cmake --build . --config Debug
```

テストは `test-assets/` ディレクトリをカレントにして実行する必要がある（テスト用音声ファイルの相対パス解決のため）。OnnxRuntimeのDLLパスは `core\third-party\onnxruntime\lib\windows\x64` を `PATH` に追加する。

### Linux/macOS

```bash
scripts/run-core-tests.sh
```

Linux実行時は `LD_LIBRARY_PATH` に `core/third-party/onnxruntime/lib/linux/x86_64` を設定する。

### 単体テストの個別実行

ビルド後、`test-assets/` から直接テスト実行ファイルを呼び出せる:

```bash
cd test-assets
../core/build/transcriber-test        # Transcriberテストのみ
../core/build/moonshine-c-api-test    # C APIテストのみ
```

主要テスト一覧: `bin-tokenizer-test`, `onnxruntime-test`, `debug-utils-test`, `string-utils-test`, `resampler-test`, `voice-activity-detector-test`, `transcriber-test`, `moonshine-c-api-test`, `moonshine-cpp-test`, `cosine-distance-test`, `speaker-embedding-model-test`, `online-clusterer-test`

### C++フォーマット

```bash
./clang-format.sh
```

`.clang-format` で定義（LLVMベース、2スペースインデント、80カラム制限）。

## アーキテクチャ

### レイヤー構成

```
言語バインディング (Python / Swift / Android JNI)
         ↓
   C API (moonshine-c-api.h) — スレッドセーフ、ハンドルベース
         ↓
   C++20 コアエンジン (core/)
         ↓
   OnnxRuntime (推論) + Silero (VAD)
```

全プラットフォーム共通のC++コアを唯一の実装とし、各言語バインディングはC APIを通じてアクセスする。

### コアの主要コンポーネント（`core/`）

- **Transcriber** (`transcriber.h/.cpp`): ストリーミング音声認識パイプライン全体を管理。VAD → ASR → テキスト出力の一連の処理を統括
- **MoonshineModel / MoonshineStreamingModel**: OnnxRuntimeベースの推論エンジン。ストリーミング版はエンコーダキャッシュによる低レイテンシ実現
- **VoiceActivityDetector** (`voice-activity-detector.h/.cpp`): Silero VADを使った音声区間検出
- **IntentRecognizer** (`intent-recognizer.h/.cpp`): Gemma 300M埋め込みモデルによるコマンド認識（コサイン類似度ベース）
- **SpeakerEmbeddingModel + OnlineClusterer**: 話者分離（ダイアライゼーション）。Pyannoteベースの話者埋め込み＋オンラインクラスタリング

### C API設計パターン

- **ハンドルベース**: `moonshine_load_transcriber_from_files()` が `int32_t` ハンドルを返し、以降の操作はこのハンドルで識別
- **ストリームモデル**: 1つのTranscriberに対して複数のStreamを作成可能（`moonshine_create_stream`）
- **メモリ所有権**: 全メモリはC++ライブラリ側が所有。返されたポインタは次回の同transcriberへの呼び出しまで有効
- **スレッドセーフ**: API呼び出しはスレッドセーフ（ただし同一transcriber内の計算はシリアライズ）

### イベント駆動トランスクリプション

ストリーミング結果は `transcript_line_t` のリストとして返され、各行に `is_new` / `has_text_changed` / `is_complete` / `is_updated` フラグがある。行は追加のみ（削除されない）、完了した行は更新されない。

### Pythonバインディング（`python/src/moonshine_voice/`）

- `moonshine_api.py`: ctypesによるC構造体バインディング
- `transcriber.py`: Transcriber本体とイベントリスナーシステム（LineStarted / LineTextChanged / LineCompleted）
- `mic_transcriber.py`: sounddeviceを使ったマイク入力キャプチャ
- `intent_recognizer.py`: 意図認識のPythonラッパー
- `download.py`: モデルファイルの自動ダウンロード
- `__init__.py`: 遅延インポートで不要な依存関係の読み込みを回避

### モデルアーキテクチャ

```
TINY (0)             → 26M params, 非ストリーミング
BASE (1)             → 58M params, 非ストリーミング
TINY_STREAMING (2)   → 34M params
BASE_STREAMING (3)   → ~120M params
SMALL_STREAMING (4)  → 123M params
MEDIUM_STREAMING (5) → 245M params
```

モデルファイル: `encoder_model.ort`, `decoder_model_merged.ort`, `tokenizer.bin`（ORT形式の量子化ONNXモデル）

### ビルドバリアント

- **共有ライブラリ**（デフォルト）: Linux, macOS, Android
- **静的ライブラリ**: iOS/Swift フレームワーク, Windows
- Windowsの共有ビルドではシンボルエクスポートの制限があり、一部テストがスキップされる

## バージョン管理

バージョンは `scripts/update-version.sh` で一括更新。変更が必要な箇所: `core/CMakeLists.txt`, `python/pyproject.toml`, `build.gradle.kts`

## コンパイラフラグ

非Windowsビルドでは `-Wall -Wextra -pedantic -Werror` が有効。`moonshine-cpp-test` は意図的にC++11でビルドし、公開APIの後方互換性を検証している。
