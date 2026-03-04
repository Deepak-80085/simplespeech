import os
import re

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_ID = os.getenv("OLLAMA_MODEL", "qwem3.5-8_bit")
SYSTEM_PROMPT = (
    "You are a transcript cleaner.\n"
    "Rewrite the text into clean written English while preserving exact meaning.\n"
    "Rules:\n"
    "- Remove filler words and discourse noise: uh, um, like, you know, i mean, basically, hmm.\n"
    "- Remove immediate word repetitions (e.g., 'test test' -> 'test').\n"
    "- Keep names, facts, and intent unchanged.\n"
    "- Fix capitalization and punctuation.\n"
    "- Output only the final cleaned text.\n"
    "- Do not add labels, quotes, explanations, or extra commentary."
)


class Refiner:
    def __init__(self):
        self._warned_unavailable = False
        print(f"Refiner ready (ollama/{MODEL_ID})")

    @staticmethod
    def _clean_output(text):
        cleaned = text.strip()

        # Some models leak hidden reasoning blocks in plain text output.
        if "<think>" in cleaned.lower():
            cleaned = cleaned.split("<think>", 1)[0].strip()
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()
        cleaned = re.sub(r"(?im)^thinking process:.*$", "", cleaned).strip()

        for prefix in ("Cleaned transcript:", "Cleaned text:", "Output:"):
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()

        # Remove markdown code fences if model wraps output.
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        return cleaned.strip().strip('"').strip("'")

    def _call_ollama(self, model, raw_transcript):
        payload = {
            "model": model,
            "prompt": raw_transcript,
            "system": SYSTEM_PROMPT,
            "stream": False,
            # Reasoning models can spend all tokens in "thinking" and return empty output.
            "think": False,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "repeat_penalty": 1.05,
                "num_predict": 96,
            },
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)

        # Backward compatibility for Ollama versions that do not support "think".
        if response.status_code == 400 and "think" in response.text.lower():
            payload.pop("think", None)
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)

        response.raise_for_status()
        return self._clean_output(response.json().get("response", ""))

    def refine(self, raw_transcript):
        if not raw_transcript or not raw_transcript.strip():
            return raw_transcript

        try:
            cleaned = self._call_ollama(MODEL_ID, raw_transcript)
            if not re.search(r"[A-Za-z0-9]", cleaned or ""):
                return raw_transcript
            return cleaned or raw_transcript
        except requests.exceptions.RequestException as exc:
            if not self._warned_unavailable:
                print(f"[Refiner warning] Ollama unavailable: {exc}")
                self._warned_unavailable = True
            return raw_transcript
