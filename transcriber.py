import inspect
import time

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self):
        print("Loading Distil-Whisper Large-v3...")
        self.model = WhisperModel(
            model_size_or_path="distil-large-v3",
            device="cuda",
            compute_type="int8_float16",
        )

        signature = inspect.signature(self.model.transcribe)
        self._supports_chunk_length_s = "chunk_length_s" in signature.parameters
        self._supports_chunk_length = "chunk_length" in signature.parameters

        print("Model loaded into GPU.")

    def transcribe(self, audio_path):
        start = time.time()

        kwargs = {
            "beam_size": 1,
            "language": "en",
            "condition_on_previous_text": False,
            "vad_filter": True,
        }

        if self._supports_chunk_length_s:
            kwargs["chunk_length_s"] = 30
        elif self._supports_chunk_length:
            kwargs["chunk_length"] = 30

        segments, _ = self.model.transcribe(audio_path, **kwargs)

        output_string = " ".join([segment.text for segment in segments]).strip()
        duration = time.time() - start
        return output_string, round(duration, 2)
