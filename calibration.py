import difflib
import itertools
import re

import jellyfish

from database import (
    add_correction,
    add_or_increment_calibration_suggestion,
    get_calibration_suggestion_pair,
    get_dictionary_terms,
    mark_calibration_suggestion_status,
    set_app_state,
)

CALIBRATION_DONE_KEY = "calibration_done"
AUTO_LEARN_MIN_REPEATS = 1
AUTO_LEARN_MIN_CONFIDENCE = 0.82
MIN_SUGGESTION_CONFIDENCE = 0.72
MIN_TOKEN_LENGTH = 3
MAX_RECORD_ATTEMPTS = 3
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "am",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "and",
    "or",
    "for",
    "with",
    "it",
    "this",
    "that",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "do",
    "did",
    "does",
    "be",
    "been",
    "have",
    "has",
    "had",
    "at",
    "by",
    "from",
    "as",
    "if",
    "but",
    "so",
    "not",
}

PROMPT_TEMPLATES = [
    "My project term is {term}.",
    "Please write {term} exactly.",
    "I often say {term} during meetings.",
    "The correct spelling is {term}.",
    "Remember this keyword: {term}.",
]


def _tokenize(text):
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", text)]


def _pair_confidence(expected, misheard):
    jw = jellyfish.jaro_winkler_similarity(expected, misheard)
    phonetic_bonus = (
        0.15
        if jellyfish.metaphone(expected) and jellyfish.metaphone(expected) == jellyfish.metaphone(misheard)
        else 0.0
    )
    return min(1.0, jw + phonetic_bonus)


def _is_safe_token(token):
    return len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS


def _extract_mismatch_pairs(expected_text, heard_text):
    expected_tokens = _tokenize(expected_text)
    heard_tokens = _tokenize(heard_text)
    matcher = difflib.SequenceMatcher(a=expected_tokens, b=heard_tokens)
    pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        expected_chunk = expected_tokens[i1:i2]
        heard_chunk = heard_tokens[j1:j2]
        for expected_token, heard_token in zip(expected_chunk, heard_chunk):
            if expected_token == heard_token:
                continue
            pairs.append((heard_token, expected_token))
    return pairs


def _build_calibration_sentences(target_count=10):
    dictionary_terms = [term for term, _, _, _ in get_dictionary_terms(enabled_only=True)]
    if not dictionary_terms:
        return []

    prompts = []
    for term, template in zip(
        itertools.cycle(dictionary_terms),
        itertools.cycle(PROMPT_TEMPLATES),
    ):
        prompts.append(template.format(term=term))
        if len(prompts) >= target_count:
            break
    return prompts


def _learn_from_pair(misheard, expected):
    if not (_is_safe_token(misheard) and _is_safe_token(expected)):
        return "skipped"

    confidence = _pair_confidence(expected, misheard)
    if confidence < MIN_SUGGESTION_CONFIDENCE:
        return "skipped"

    add_or_increment_calibration_suggestion(
        misheard,
        expected,
        confidence=confidence,
        status="pending",
    )

    pair_row = get_calibration_suggestion_pair(misheard, expected)
    if pair_row is None:
        return "skipped"

    suggestion_id, _, _, _, seen_count, _ = pair_row
    if seen_count >= AUTO_LEARN_MIN_REPEATS and confidence >= AUTO_LEARN_MIN_CONFIDENCE:
        add_correction(misheard, expected, source="calibration")
        mark_calibration_suggestion_status(suggestion_id, "accepted")
        return "accepted"
    return "pending"


def run_calibration(transcriber, record_audio_fn, cleanup_audio_fn, rounds=10):
    print("\n--- Calibration ---")
    print("Read each sentence naturally after pressing Enter.")

    prompts = _build_calibration_sentences(target_count=rounds)
    if not prompts:
        set_app_state(CALIBRATION_DONE_KEY, "false")
        print("No dictionary terms found. Add terms first (A), then run calibration (K).")
        return False

    accepted = 0
    pending = 0
    skipped = 0

    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n[{idx}/{len(prompts)}] Say: {prompt}")
        heard_text = ""
        captured = False
        for attempt in range(1, MAX_RECORD_ATTEMPTS + 1):
            input("Press Enter to record...")
            audio_path = None
            try:
                audio_path = record_audio_fn()
                heard_text, _ = transcriber.transcribe(audio_path)
                captured = True
                break
            except RuntimeError as exc:
                if "No audio captured" in str(exc):
                    print(
                        "No audio captured. Hold Right Alt for at least ~1 second and retry."
                    )
                    if attempt < MAX_RECORD_ATTEMPTS:
                        continue
                    print("Skipping this calibration sentence.")
                    break
                raise
            finally:
                cleanup_audio_fn(audio_path)

        if not captured:
            skipped += 1
            continue

        print(f"Heard: {heard_text}")
        mismatch_pairs = _extract_mismatch_pairs(prompt, heard_text)
        if not mismatch_pairs:
            print("No mismatches found.")
            continue

        for misheard, expected in mismatch_pairs:
            outcome = _learn_from_pair(misheard, expected)
            if outcome == "accepted":
                accepted += 1
            elif outcome == "pending":
                pending += 1
            else:
                skipped += 1

    set_app_state(CALIBRATION_DONE_KEY, "true")
    print(
        "\nCalibration complete. "
        f"Accepted: {accepted}, Pending review: {pending}, Skipped: {skipped}."
    )
    return True
