# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Moonshine Voice — リアルタイム音声アプリケーション向けのオンデバイス音声認識(STT)ライブラリ。C++20コアに Python / Swift / Android(JNI) のバインディングを持つクロスプラットフォーム構成。8言語以上対応（英語はMITライセンス、他言語はCommunity License）。

## ビルドとテスト

### C++コアのビルド＆テスト（Windows）

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

### C++コアのビルド＆テスト（Linux/macOS）

```bash
scripts/run-core-tests.sh
```

Linux実行時は `LD_LIBRARY_PATH` に `core/third-party/onnxruntime/lib/linux/x86_64` を設定する。

### 単体テストの個別実行

ビルド後、`test-assets/` から直接テスト実行ファイルを呼び出せる:

```bash
cd test-assets
../core/build/transcriber-test        # 例: Transcriberテストのみ
../core/build/moonshine-c-api-test    # C APIテストのみ
```

主要テスト一覧: `bin-tokenizer-test`, `onnxruntime-test`, `debug-utils-test`, `string-utils-test`, `resampler-test`, `voice-activity-detector-test`, `transcriber-test`, `moonshine-c-api-test`, `moonshine-cpp-test`, `cosine-distance-test`, `speaker-embedding-model-test`, `online-clusterer-test`

### C++フォーマット

```bash
./clang-format.sh
```

`.clang-format` で定義（LLVMベース、2スペースインデント、80カラム制限）。

### Pythonパッケージ

```bash
cd python && pip install -e .
```

`python/src/moonshine_voice/` にソースがある。ネイティブライブラリ（`.dylib`/`.so`/`.dll`）を同梱したプラットフォーム固有wheelとして配布。

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
