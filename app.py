import argparse
import ctypes
import os
import platform
import queue
import re
import threading
import time
import uuid

import jellyfish
import numpy as np
import pyperclip
import sounddevice as sd
from pynput import keyboard
from scipy.io.wavfile import write

from calibration import CALIBRATION_DONE_KEY, run_calibration
from database import (
    add_correction,
    add_dictionary_term,
    get_app_state,
    get_corrections,
    get_dictionary_terms,
    increment_dictionary_usage,
    init_db,
    remove_dictionary_term,
)
from refiner import Refiner
from transcriber import Transcriber

SAMPLE_RATE = 16000
FILENAME = "temp_recording.wav"
POLL_INTERVAL_S = 0.01
PASTE_DELAY_S = 0.04
INDICATOR_POLL_MS = 50
INDICATOR_DEFAULT_TTL_S = 1.2
ALT_HOLD_TRIGGER_S = 0.28
PRINT_HOTKEY_DEBUG_TRANSCRIPTS = True

IS_WINDOWS = os.name == "nt"
IS_MACOS = platform.system() == "Darwin"

VK_RMENU = 0xA5
_user32 = ctypes.windll.user32 if IS_WINDOWS else None

ALT_KEYS = tuple(
    key
    for key in (
        keyboard.Key.alt,
        keyboard.Key.alt_l,
        keyboard.Key.alt_r,
        getattr(keyboard.Key, "alt_gr", None),
    )
    if key is not None
)
SHIFT_KEYS = tuple(
    key
    for key in (
        keyboard.Key.shift,
        keyboard.Key.shift_l,
        keyboard.Key.shift_r,
    )
    if key is not None
)


def is_right_alt_pressed():
    if _user32 is None:
        return False
    return bool(_user32.GetAsyncKeyState(VK_RMENU) & 0x8000)


def record_audio():
    if not IS_WINDOWS:
        raise RuntimeError("Right Alt hold recording is only supported on Windows.")
    print("\nHold Right Alt to record. Release Right Alt to stop.")
    while not is_right_alt_pressed():
        time.sleep(POLL_INTERVAL_S)
    print("Recording...")
    chunks = []

    def callback(indata, _frames, _time_info, _status):
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        while is_right_alt_pressed():
            time.sleep(POLL_INTERVAL_S)

    if not chunks:
        raise RuntimeError("No audio captured.")

    audio = np.concatenate(chunks, axis=0)
    write(FILENAME, SAMPLE_RATE, audio)
    print(f"Audio captured ({audio.shape[0] / SAMPLE_RATE:.2f}s).")
    return FILENAME


def cleanup_audio_file(path):
    if path and os.path.exists(path):
        os.remove(path)


def _match_word_token(token):
    return re.match(r"^([^A-Za-z0-9']*)([A-Za-z0-9']+)([^A-Za-z0-9']*)$", token)


def _preserve_case(original_word, replacement_word):
    if original_word.isupper():
        return replacement_word.upper()
    if original_word and original_word[0].isupper():
        return replacement_word[:1].upper() + replacement_word[1:]
    return replacement_word


def apply_corrections(transcript):
    corrections = get_corrections()
    dictionary_terms = get_dictionary_terms(enabled_only=True)

    exact_corrections = {}
    phonetic_corrections = {}
    for original, corrected, phonetic in corrections:
        key = original.lower()
        exact_corrections[key] = corrected
        if phonetic and phonetic not in phonetic_corrections:
            phonetic_corrections[phonetic] = corrected

    dictionary_exact = {}
    dictionary_phonetic = {}
    for term, phonetic, _, _ in dictionary_terms:
        term_key = term.lower()
        dictionary_exact[term_key] = term
        if phonetic and phonetic not in dictionary_phonetic:
            dictionary_phonetic[phonetic] = term

    output_tokens = []
    for token in transcript.split():
        match = _match_word_token(token)
        if not match:
            output_tokens.append(token)
            continue

        prefix, base_word, suffix = match.groups()
        clean_word = base_word.lower()
        word_phonetic = jellyfish.metaphone(clean_word) if clean_word else ""
        replacement = None
        dictionary_hit = None

        if clean_word in exact_corrections:
            replacement = exact_corrections[clean_word]
        elif word_phonetic and word_phonetic in phonetic_corrections:
            replacement = phonetic_corrections[word_phonetic]
        elif clean_word in dictionary_exact:
            replacement = dictionary_exact[clean_word]
            dictionary_hit = dictionary_exact[clean_word]
        elif word_phonetic and word_phonetic in dictionary_phonetic:
            replacement = dictionary_phonetic[word_phonetic]
            dictionary_hit = dictionary_phonetic[word_phonetic]

        if replacement is None:
            output_tokens.append(token)
            continue

        replacement = _preserve_case(base_word, replacement)
        output_tokens.append(f"{prefix}{replacement}{suffix}")

        if dictionary_hit is not None:
            increment_dictionary_usage(dictionary_hit)

    return " ".join(output_tokens)


def print_dictionary():
    terms = get_dictionary_terms(enabled_only=False)
    if not terms:
        print("Dictionary is empty.")
        return

    print("\n--- Dictionary Terms ---")
    for idx, (term, phonetic, enabled, usage_count) in enumerate(terms, start=1):
        status = "enabled" if enabled else "disabled"
        print(
            f"{idx}. {term} | phonetic={phonetic or '-'} | "
            f"{status} | used={usage_count}"
        )


def handle_dictionary_add():
    term = input("Add word: ").strip()
    if not term:
        print("No word entered.")
        return
    add_dictionary_term(term)
    print(f"Added '{term}' to dictionary.")


def handle_dictionary_remove():
    term = input("Remove word: ").strip()
    if not term:
        print("No word entered.")
        return
    if remove_dictionary_term(term):
        print(f"Removed '{term}' from dictionary.")
    else:
        print(f"'{term}' not found in dictionary.")


def run_calibration_safe(transcriber):
    try:
        completed = run_calibration(transcriber, record_audio, cleanup_audio_file)
        if completed is False:
            print("Calibration not completed.")
    except RuntimeError as exc:
        print(f"Calibration failed: {exc}")
    except Exception as exc:
        print(f"Unexpected calibration error: {exc}")


def paste_text_at_cursor(text, key_controller):
    payload = text.strip()
    if not payload:
        return False

    previous_clipboard = None
    should_restore_clipboard = False
    try:
        previous_clipboard = pyperclip.paste()
        should_restore_clipboard = True
    except pyperclip.PyperclipException:
        should_restore_clipboard = False

    pyperclip.copy(payload)
    time.sleep(PASTE_DELAY_S)

    modifier_key = keyboard.Key.cmd if IS_MACOS else keyboard.Key.ctrl
    with key_controller.pressed(modifier_key):
        key_controller.press("v")
        key_controller.release("v")

    if should_restore_clipboard:
        time.sleep(PASTE_DELAY_S)
        try:
            pyperclip.copy(previous_clipboard)
        except pyperclip.PyperclipException:
            pass

    return True


class FloatingStatusIndicator:
    COLORS = {
        "recording": {"bg": "#8B1E1E", "fg": "#FFF4F4"},
        "working": {"bg": "#1F3044", "fg": "#EAF3FF"},
        "success": {"bg": "#1E4D2B", "fg": "#EFFFF1"},
        "warning": {"bg": "#5A4200", "fg": "#FFF5D6"},
        "error": {"bg": "#6A1B1B", "fg": "#FFECEC"},
        "info": {"bg": "#2B2B2B", "fg": "#F5F5F5"},
    }

    def __init__(self):
        self._tk = None
        self._root = None
        self._frame = None
        self._label = None
        self._commands = queue.Queue()
        self._hide_deadline = None
        self._started = False
        self._available = True

    def start(self):
        if self._started:
            return
        try:
            import tkinter as tk
        except Exception:
            self._available = False
            print("[SimpleSpeech] Floating indicator unavailable on this environment.")
            return

        self._tk = tk
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        try:
            self._root.attributes("-alpha", 0.95)
        except Exception:
            pass

        self._frame = tk.Frame(self._root, bg=self.COLORS["info"]["bg"], bd=0, padx=14, pady=9)
        self._frame.pack(fill="both", expand=True)
        self._label = tk.Label(
            self._frame,
            text="",
            bg=self.COLORS["info"]["bg"],
            fg=self.COLORS["info"]["fg"],
            font=("Segoe UI", 10, "bold"),
        )
        self._label.pack()
        self._started = True

    def stop(self):
        if not self._started or not self._available:
            return
        try:
            self._root.destroy()
        except Exception:
            pass
        self._root = None
        self._frame = None
        self._label = None
        self._started = False
        self._hide_deadline = None

    def show(self, text, tone="info", ttl=None):
        if not self._available or not self._started:
            return
        self._commands.put(("show", text, tone, ttl))

    def hide(self):
        if not self._available or not self._started:
            return
        self._commands.put(("hide",))

    def _place_window(self):
        self._root.update_idletasks()
        width = self._root.winfo_reqwidth()
        height = self._root.winfo_reqheight()
        x = max(0, (self._root.winfo_screenwidth() - width) // 2)
        y = max(20, int(self._root.winfo_screenheight() * 0.08))
        self._root.geometry(f"{width}x{height}+{x}+{y}")

    def _apply_show(self, text, tone, ttl):
        style = self.COLORS.get(tone, self.COLORS["info"])
        self._frame.configure(bg=style["bg"])
        self._label.configure(text=text, bg=style["bg"], fg=style["fg"])
        self._place_window()
        self._root.deiconify()
        self._root.lift()
        if ttl is not None and ttl > 0:
            self._hide_deadline = time.time() + ttl
        else:
            self._hide_deadline = None

    def _apply_hide(self):
        self._hide_deadline = None
        self._root.withdraw()

    def process_events(self):
        if not self._available or not self._started:
            return

        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                break

            action = cmd[0]
            if action == "show":
                _, text, tone, ttl = cmd
                self._apply_show(text, tone, ttl)
            elif action == "hide":
                self._apply_hide()

        if self._hide_deadline is not None and time.time() >= self._hide_deadline:
            self._apply_hide()

        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._available = False
            self._started = False


class HotkeyAudioRecorder:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._stream = None
        self._chunks = []

    def _callback(self, indata, _frames, _time_info, _status):
        with self._lock:
            if self._stream is not None:
                self._chunks.append(indata.copy())

    def start(self):
        with self._lock:
            if self._stream is not None:
                return False
            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=self._callback,
            )
            self._stream.start()
            return True

    def stop_and_save(self):
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is None:
            raise RuntimeError("No active recording.")

        stream.stop()
        stream.close()

        with self._lock:
            chunks = self._chunks
            self._chunks = []

        if not chunks:
            raise RuntimeError("No audio captured.")

        audio = np.concatenate(chunks, axis=0)
        filename = f"temp_recording_{uuid.uuid4().hex}.wav"
        write(filename, self.sample_rate, audio)
        duration = audio.shape[0] / self.sample_rate
        return filename, duration

    def abort(self):
        with self._lock:
            stream = self._stream
            self._stream = None
            self._chunks = []
        if stream is None:
            return
        stream.stop()
        stream.close()


class HotkeyDictationService:
    def __init__(self, transcriber, refiner, indicator=None):
        self._transcriber = transcriber
        self._refiner = refiner
        self._indicator = indicator
        self._recorder = HotkeyAudioRecorder()
        self._paste_controller = keyboard.Controller()

        self._state_lock = threading.Lock()
        self._alt_down = set()
        self._shift_down = set()
        self._recording = False
        self._refine_requested = False
        self._alt_started_at = None
        self._ignore_alt_cycle = False

        self._jobs = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._listener = None
        self._running = False

    def start(self):
        self._running = True
        self._worker.start()
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.start()

    def stop(self):
        self._running = False
        if self._listener is not None:
            self._listener.stop()
        self._recorder.abort()
        if self._indicator is not None:
            self._indicator.hide()
        self._jobs.put(None)
        self._worker.join(timeout=5)

    def _on_press(self, key):
        with self._state_lock:
            if key in ALT_KEYS:
                if not self._alt_down and not self._recording:
                    self._alt_started_at = time.monotonic()
                    self._ignore_alt_cycle = False
                self._alt_down.add(key)
            if key in SHIFT_KEYS:
                self._shift_down.add(key)

            if self._recording and self._shift_down:
                self._refine_requested = True

            # If Alt is being used with any non-modifier key, ignore this Alt cycle.
            if self._alt_down and not self._recording:
                if key not in ALT_KEYS and key not in SHIFT_KEYS:
                    self._ignore_alt_cycle = True

    def _on_release(self, key):
        should_stop_recording = False
        mode = "raw"

        with self._state_lock:
            if key in ALT_KEYS:
                self._alt_down.discard(key)
            if key in SHIFT_KEYS:
                self._shift_down.discard(key)

            if self._recording and not self._alt_down:
                should_stop_recording = True
                mode = "refined" if self._refine_requested else "raw"
                self._recording = False
                self._refine_requested = False
            elif not self._recording and not self._alt_down:
                self._alt_started_at = None
                self._ignore_alt_cycle = False

        if not should_stop_recording:
            return

        try:
            audio_path, audio_len = self._recorder.stop_and_save()
            self._jobs.put((audio_path, mode))
            print(f"[SimpleSpeech] Captured {audio_len:.2f}s. Processing ({mode})...")
            if self._indicator is not None:
                self._indicator.show("Transcribing...", tone="working")
        except Exception as exc:
            print(f"[SimpleSpeech] Could not stop recording: {exc}")
            if self._indicator is not None:
                self._indicator.show("Capture failed", tone="error", ttl=2.0)

    def tick(self):
        should_start_recording = False

        with self._state_lock:
            if self._recording:
                return
            if not self._alt_down:
                return
            if self._ignore_alt_cycle:
                return
            if self._alt_started_at is None:
                self._alt_started_at = time.monotonic()
                return
            if time.monotonic() - self._alt_started_at < ALT_HOLD_TRIGGER_S:
                return

            self._recording = True
            self._refine_requested = bool(self._shift_down)
            should_start_recording = True

        if not should_start_recording:
            return

        try:
            started = self._recorder.start()
            if not started:
                with self._state_lock:
                    self._recording = False
                return
            mode_label = "refined" if self._refine_requested else "raw"
            print(f"\n[SimpleSpeech] Recording started ({mode_label}). Release Alt to stop.")
            if self._indicator is not None:
                self._indicator.show(f"Recording ({mode_label})", tone="recording")
        except Exception as exc:
            with self._state_lock:
                self._recording = False
                self._refine_requested = False
                self._alt_started_at = None
            print(f"[SimpleSpeech] Could not start recording: {exc}")
            if self._indicator is not None:
                self._indicator.show("Mic error", tone="error", ttl=2.0)

    def _worker_loop(self):
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                break

            audio_path, mode = job
            try:
                raw, duration = self._transcriber.transcribe(audio_path)
                corrected = apply_corrections(raw)

                refined_text = corrected
                if mode == "refined" or PRINT_HOTKEY_DEBUG_TRANSCRIPTS:
                    if self._indicator is not None:
                        if mode == "refined":
                            self._indicator.show("Refining...", tone="working")
                    refined_text = self._refiner.refine(corrected)

                if PRINT_HOTKEY_DEBUG_TRANSCRIPTS:
                    print("\n--- HOTKEY RAW ---")
                    print(raw)
                    print("\n--- HOTKEY REFINED ---")
                    print(refined_text)
                    print("--------------------")

                final_text = corrected if mode == "raw" else refined_text

                pasted = paste_text_at_cursor(final_text, self._paste_controller)
                if pasted:
                    print(
                        f"[SimpleSpeech] Pasted {mode} transcript "
                        f"({len(final_text)} chars, transcribe {duration:.2f}s)."
                    )
                    if self._indicator is not None:
                        self._indicator.show(
                            f"Pasted ({mode})",
                            tone="success",
                            ttl=INDICATOR_DEFAULT_TTL_S,
                        )
                else:
                    print("[SimpleSpeech] Transcript was empty. Nothing pasted.")
                    if self._indicator is not None:
                        self._indicator.show(
                            "No speech detected",
                            tone="warning",
                            ttl=INDICATOR_DEFAULT_TTL_S,
                        )
            except Exception as exc:
                print(f"[SimpleSpeech] Processing failed: {exc}")
                if self._indicator is not None:
                    self._indicator.show("Processing error", tone="error", ttl=2.0)
            finally:
                cleanup_audio_file(audio_path)
                self._jobs.task_done()


def run_cli_mode():
    print("\n--- SimpleSpeech (CLI) ---")
    init_db()

    transcriber = Transcriber()
    refiner = Refiner()

    if get_app_state(CALIBRATION_DONE_KEY, "false") != "true":
        run_now = input(
            "Calibration has not been run yet. Run calibration now? (y/n): "
        ).strip().lower()
        if run_now == "y":
            run_calibration_safe(transcriber)

    while True:
        audio_path = None
        try:
            cmd = input(
                "\nAction [Enter=record, A=add word, D=view dict, R=remove word, "
                "K=calibrate, Q=quit]: "
            ).strip().lower()

            if cmd == "q":
                print("Exiting.")
                break
            if cmd == "a":
                handle_dictionary_add()
                continue
            if cmd == "d":
                print_dictionary()
                continue
            if cmd == "r":
                handle_dictionary_remove()
                continue
            if cmd == "k":
                run_calibration_safe(transcriber)
                continue
            if cmd not in ("", "e"):
                print("Unknown command. Use Enter/A/D/R/K/Q.")
                continue

            print("\nWaiting for Right Alt hold (Ctrl+C to exit)...")
            audio_path = record_audio()

            print("\nTranscribing...")
            raw, duration = transcriber.transcribe(audio_path)

            corrected = apply_corrections(raw)

            print("Refining...")
            refined = refiner.refine(corrected)

            print("\n--- RAW ---")
            print(raw)
            print("\n--- REFINED ---")
            print(refined)
            print(f"------------------ (took {duration}s)")

            feedback = input("\nDid Whisper mishear a word? (y/n): ").strip().lower()
            if feedback == "y":
                wrong = input("What did it output (wrong word)? ").strip()
                right = input("What should it be? ").strip()
                add_correction(wrong, right, source="manual")
                print(f"Learned: '{wrong}' -> '{right}'")
            else:
                print("Good.")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        finally:
            cleanup_audio_file(audio_path)


def run_hotkey_mode():
    print("\n--- SimpleSpeech (Hotkey Core) ---")
    print(f"Hold Alt (~{ALT_HOLD_TRIGGER_S:.2f}s) = raw transcript paste")
    print(f"Hold Alt+Shift (~{ALT_HOLD_TRIGGER_S:.2f}s) = refined transcript paste")
    print("Works with left or right Alt/Shift. Press Ctrl+C to quit.\n")

    init_db()
    transcriber = Transcriber()
    refiner = Refiner()

    if get_app_state(CALIBRATION_DONE_KEY, "false") != "true":
        print(
            "[SimpleSpeech] Calibration not completed yet. "
            "Run `python app.py --cli` then press K to calibrate."
        )

    indicator = FloatingStatusIndicator()
    indicator.start()

    service = HotkeyDictationService(transcriber, refiner, indicator=indicator)
    service.start()

    try:
        while True:
            service.tick()
            indicator.process_events()
            time.sleep(INDICATOR_POLL_MS / 1000.0)
    except KeyboardInterrupt:
        print("\n[SimpleSpeech] Exiting.")
    finally:
        service.stop()
        indicator.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="SimpleSpeech")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run legacy CLI mode instead of hotkey core mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cli:
        run_cli_mode()
        return
    run_hotkey_mode()


if __name__ == "__main__":
    main()
