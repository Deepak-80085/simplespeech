import sqlite3

import jellyfish

DB_PATH = "simplespeech_memory.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def _table_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def init_db():
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original TEXT UNIQUE NOT NULL,
            corrected TEXT NOT NULL,
            phonetic TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            source TEXT DEFAULT 'manual',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    correction_columns = _table_columns(cursor, "corrections")
    if "phonetic" not in correction_columns:
        cursor.execute("ALTER TABLE corrections ADD COLUMN phonetic TEXT DEFAULT ''")
    if "frequency" not in correction_columns:
        cursor.execute("ALTER TABLE corrections ADD COLUMN frequency INTEGER DEFAULT 1")
    if "source" not in correction_columns:
        cursor.execute("ALTER TABLE corrections ADD COLUMN source TEXT DEFAULT 'manual'")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dictionary_terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT UNIQUE NOT NULL,
            phonetic TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            usage_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            misheard TEXT NOT NULL,
            expected TEXT NOT NULL,
            phonetic_misheard TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            seen_count INTEGER DEFAULT 1,
            status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(misheard, expected)
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_corrections_original
        ON corrections(original)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_corrections_phonetic
        ON corrections(phonetic)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dictionary_terms_term
        ON dictionary_terms(term)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dictionary_terms_phonetic
        ON dictionary_terms(phonetic)
        """
    )

    conn.commit()
    conn.close()


def add_correction(original, corrected, source="manual"):
    original = original.strip()
    corrected = corrected.strip()
    phonetic = jellyfish.metaphone(original.lower())
    conn = _connect()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO corrections (original, corrected, phonetic, source)
            VALUES (?, ?, ?, ?)
            """,
            (original, corrected, phonetic, source),
        )
    except sqlite3.IntegrityError:
        cursor.execute(
            """
            UPDATE corrections
            SET frequency = frequency + 1, corrected = ?, phonetic = ?, source = ?
            WHERE original = ?
            """,
            (corrected, phonetic, source, original),
        )
    conn.commit()
    conn.close()


def get_corrections():
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT original, corrected, phonetic
        FROM corrections
        ORDER BY frequency DESC, id DESC
        """
    )
    results = cursor.fetchall()
    conn.close()
    return results


def add_dictionary_term(term):
    clean_term = term.strip()
    phonetic = jellyfish.metaphone(clean_term.lower())
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO dictionary_terms (term, phonetic, enabled)
        VALUES (?, ?, 1)
        ON CONFLICT(term) DO UPDATE SET
            phonetic = excluded.phonetic,
            enabled = 1
        """,
        (clean_term, phonetic),
    )
    conn.commit()
    conn.close()


def get_dictionary_terms(enabled_only=True):
    conn = _connect()
    cursor = conn.cursor()
    if enabled_only:
        cursor.execute(
            """
            SELECT term, phonetic, enabled, usage_count
            FROM dictionary_terms
            WHERE enabled = 1
            ORDER BY usage_count DESC, term ASC
            """
        )
    else:
        cursor.execute(
            """
            SELECT term, phonetic, enabled, usage_count
            FROM dictionary_terms
            ORDER BY enabled DESC, usage_count DESC, term ASC
            """
        )
    results = cursor.fetchall()
    conn.close()
    return results


def remove_dictionary_term(term):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dictionary_terms WHERE term = ?", (term.strip(),))
    removed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return removed


def increment_dictionary_usage(term):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE dictionary_terms
        SET usage_count = usage_count + 1
        WHERE term = ?
        """,
        (term,),
    )
    conn.commit()
    conn.close()


def get_app_state(key, default=None):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM app_state WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return default
    return row[0]


def set_app_state(key, value):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO app_state(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def add_or_increment_calibration_suggestion(
    misheard, expected, confidence=0.0, status="pending"
):
    clean_misheard = misheard.strip()
    clean_expected = expected.strip()
    phonetic = jellyfish.metaphone(clean_misheard.lower())

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO calibration_suggestions
            (misheard, expected, phonetic_misheard, confidence, seen_count, status)
        VALUES (?, ?, ?, ?, 1, ?)
        ON CONFLICT(misheard, expected) DO UPDATE SET
            seen_count = seen_count + 1,
            confidence = MAX(confidence, excluded.confidence),
            status = excluded.status
        """,
        (clean_misheard, clean_expected, phonetic, confidence, status),
    )
    conn.commit()
    conn.close()


def get_calibration_suggestions(status=None):
    conn = _connect()
    cursor = conn.cursor()
    if status is None:
        cursor.execute(
            """
            SELECT id, misheard, expected, confidence, seen_count, status
            FROM calibration_suggestions
            ORDER BY seen_count DESC, confidence DESC, id DESC
            """
        )
    else:
        cursor.execute(
            """
            SELECT id, misheard, expected, confidence, seen_count, status
            FROM calibration_suggestions
            WHERE status = ?
            ORDER BY seen_count DESC, confidence DESC, id DESC
            """,
            (status,),
        )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_calibration_suggestion_pair(misheard, expected):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, misheard, expected, confidence, seen_count, status
        FROM calibration_suggestions
        WHERE misheard = ? AND expected = ?
        """,
        (misheard.strip(), expected.strip()),
    )
    row = cursor.fetchone()
    conn.close()
    return row


def mark_calibration_suggestion_status(suggestion_id, status):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE calibration_suggestions SET status = ? WHERE id = ?",
        (status, suggestion_id),
    )
    conn.commit()
    conn.close()
