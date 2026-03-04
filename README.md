# SimpleSpeech

Offline voice-to-text tool (Python) with:
- local transcription (`faster-whisper`)
- optional local refinement (`Ollama`)
- adaptive correction memory (`SQLite`)
- global hotkeys for quick paste

## Current Features
- Hold `Alt` (left or right) to record and paste **raw** output
- Hold `Alt + Shift` to record and paste **refined** output
- Global paste into current cursor target
- Dictionary + correction memory in `simplespeech_memory.db`
- Calibration workflow (CLI mode)

## Requirements
- Python 3.11+
- Windows recommended (current input flow is optimized/tested there)
- Microphone access
- GPU recommended for faster transcription
- Ollama running locally if you want refinement

Install deps:

```powershell
.\venv\Scripts\pip install -r requirements.txt
```

## Run
Default hotkey mode:

```powershell
.\venv\Scripts\python app.py
```

Legacy CLI mode:

```powershell
.\venv\Scripts\python app.py --cli
```

## Hotkey Behavior
- Recording starts after holding Alt for ~`0.28s`
- If Alt is used with other non-modifier keys (for normal shortcuts), that Alt cycle is ignored
- Release Alt to stop and process

## Ollama Refiner
`refiner.py` calls local Ollama endpoint:
- URL: `http://localhost:11434/api/generate` (default)
- Model env var: `OLLAMA_MODEL`

Example:

```powershell
$env:OLLAMA_MODEL="qwen3.5-8bit-fixed"
.\venv\Scripts\python app.py
```

If Ollama is unavailable or output is invalid, app falls back to raw text.

## Project Files
- `app.py` - main app (hotkey + CLI flows)
- `transcriber.py` - whisper wrapper
- `refiner.py` - Ollama refinement logic
- `database.py` - SQLite schema + persistence
- `calibration.py` - calibration and auto-learn logic
- `requirements.txt` - minimal pinned runtime deps

## Notes
- Current hotkey mode prints RAW/REFINED text in terminal for testing (`PRINT_HOTKEY_DEBUG_TRANSCRIPTS` in `app.py`).
- Calibration and dictionary management are currently easiest through `--cli`.
