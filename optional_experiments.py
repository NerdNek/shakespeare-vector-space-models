"""Optional experiments for Assignment 3 — separate from required outputs.

Each experiment states what it measures and what it cannot claim.  Results are
compared descriptively to the baseline tokenization / weighting; we do not treat
a different ranking as "more accurate" without external evidence.

Run after the required ``main.py`` path works:

    python optional_experiments.py
    python optional_experiments.py --word throne
    python optional_experiments.py --characters --limit-chars 20
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict

from main import (
    _input_path,
    _top_neighbors,
    compute_cosine_similarity,
    create_PPMI_matrix_sparse,
    create_term_context_matrix_sparse,
    create_term_document_matrix,
    create_tf_idf_matrix,
    get_column_vector,
    rank_plays,
    rank_words,
    read_in_shakespeare,
)


# Minimal English stop-word list for an optional ablation only.
_STOP_WORDS = frozenset(
    """
    a an the and or but if in on at to for of as by with from is are was were
    be been being it this that these those i you he she we they my your his her
    their our me him them not no nor so than then too very can will just
    """.split()
)


def explore_word_neighbors(word, vocab, td_matrix, tc_matrix, ppmi_matrix, limit=10):
    """Compare nearest neighbors of one word across three representations.

    Measures: which other vocabulary rows have the most similar usage profile
    under (1) term-document counts, (2) raw term-context counts, and (3) PPMI.
    Limitation: "nearest" is only with respect to the chosen similarity
    (cosine here).  Agreement across matrices is interesting but not proof of
    synonymy; Shakespeare spelling and character names dominate many lists.
    """
    word_to_id = {w: i for i, w in enumerate(vocab)}
    if word not in word_to_id:
        print(f"Word {word!r} is not in vocab.txt — try another token.")
        return
    target_id = word_to_id[word]
    print(f"\n=== Word explorer for {word!r} (cosine, top {limit}, target excluded) ===")
    for label, matrix in (
        ("term-document counts", td_matrix),
        ("term-context counts", tc_matrix),
        ("PPMI", ppmi_matrix),
    ):
        ranked = rank_words(target_id, matrix, compute_cosine_similarity)
        neighbors = _top_neighbors(ranked, target_id, limit=limit)
        print(f"\n{label}:")
        for rank, word_id in enumerate(neighbors, start=1):
            print(f"  {rank}: {vocab[word_id]}")


def compare_play_rankings_raw_vs_tfidf(document_names, td_matrix, tf_idf_matrix, target_play):
    """Side-by-side play neighbors under raw counts vs TF-IDF (cosine).

    Measures: how inverse-document weighting reorders play similarity relative
    to raw frequency vectors.  Limitation: a change in order is not automatically
    an improvement; TF-IDF down-weights words shared by many plays, which may
    help or hurt thematic grouping depending on the target.
    """
    print(
        f'\n=== Play ranking: raw counts vs TF-IDF for '
        f'"{document_names[target_play]}" (cosine) ==='
    )
    raw = _top_neighbors(
        rank_plays(target_play, td_matrix, compute_cosine_similarity),
        target_play,
        limit=10,
    )
    weighted = _top_neighbors(
        rank_plays(target_play, tf_idf_matrix, compute_cosine_similarity),
        target_play,
        limit=10,
    )
    print(f"{'rank':<5}{'raw counts':<32}{'TF-IDF':<32}")
    for rank, (r, w) in enumerate(zip(raw, weighted), start=1):
        print(f"{rank:<5}{document_names[r]:<32}{document_names[w]:<32}")
    shared = set(raw) & set(weighted)
    print(
        f"Overlap in top-10 neighbor sets: {len(shared)} / 10 "
        "(descriptive only; not an accuracy score)."
    )


def read_character_lines():
    """Return ``(character_name, tokens)`` for rows that name a speaker.

    Uses CSV column index 4 (assignment suggestion).  Stage directions and
    empty speakers are skipped.  Limitation: some names collide across plays
    (e.g., multiple characters called \"Messenger\").
    """
    lines = []
    with _input_path("will_play_text.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        reader = csv.reader(source, delimiter=";")
        for row in reader:
            if len(row) <= 5:
                continue
            speaker = row[4].strip()
            if not speaker:
                continue
            tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", row[5]).lower().split()
            if tokens:
                lines.append((speaker, tokens))
    return lines


def compare_characters(vocab, limit_chars=25, window=2):
    """Build a term-character count matrix and list cosine neighbors.

    Measures: which speakers have similar bags of vocabulary words in their
    dialogue.  Limitation: line counts vary wildly; long-speaking characters
    dominate length before cosine normalization.  Name collisions across plays
    merge distinct people.  This is exploratory, not a character-identity test.
    """
    char_lines = read_character_lines()
    # Keep the most frequent speakers so the optional run stays tractable.
    counts = defaultdict(int)
    for speaker, tokens in char_lines:
        counts[speaker] += len(tokens)
    character_names = [
        name
        for name, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[
            :limit_chars
        ]
    ]
    print(
        f"\n=== Character comparison (top {len(character_names)} speakers by "
        f"token count; term-character counts + cosine) ==="
    )
    td = create_term_document_matrix(char_lines, character_names, vocab)
    # Self-check: a character should have cosine 1 with itself when nonzero.
    for index, name in enumerate(character_names[:3]):
        vec = get_column_vector(td, index)
        self_sim = compute_cosine_similarity(vec, vec)
        print(f"Self-cosine({name!r}) = {self_sim:.4f}")

    target = 0
    ranked = rank_plays(target, td, compute_cosine_similarity)
    neighbors = _top_neighbors(ranked, target, limit=10)
    print(f'\nNearest speakers to "{character_names[target]}":')
    for rank, idx in enumerate(neighbors, start=1):
        print(f"  {rank}: {character_names[idx]}")
    if len(character_names) > 1:
        farthest = ranked[-1]
        if farthest == target and len(ranked) > 1:
            farthest = ranked[-2]
        print(
            f'Most distant among this subset: "{character_names[farthest]}" '
            f"(still only within the {limit_chars}-speaker subset)."
        )


def stopword_stem_ablation(lines, document_names, vocab, target_play):
    """Optional preprocessing variants vs the assignment's original tokenization.

    Measures: whether dropping stop words (and a tiny suffix stemmer) changes
    play rankings relative to the baseline.  Limitation: the crude stemmer is
    not Porter; it can conflate unrelated forms.  Do not claim better accuracy
    — only report whether neighbor lists move.

    Stop-word removal still uses the assignment ``vocab`` (tokens simply omit
    stops).  Stemming rebuilds a vocabulary from stemmed tokens; otherwise
    stemmed forms absent from the original ``vocab.txt`` would be dropped and
    the ablation would mostly measure data loss, not merging of word forms.
    """

    def strip_stops(line_tuples):
        return [
            (play, [tok for tok in tokens if tok not in _STOP_WORDS])
            for play, tokens in line_tuples
        ]

    def crude_stem(token):
        for suffix in ("ing", "ed", "es", "s"):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    def stem_tokens(line_tuples):
        return [
            (play, [crude_stem(tok) for tok in tokens])
            for play, tokens in line_tuples
        ]

    def matrix_with_local_vocab(line_tuples):
        local_vocab = sorted({tok for _, tokens in line_tuples for tok in tokens})
        return create_term_document_matrix(line_tuples, document_names, local_vocab)

    baseline = create_term_document_matrix(lines, document_names, vocab)
    stopped = create_term_document_matrix(
        strip_stops(lines), document_names, vocab
    )
    stemmed_lines = stem_tokens(lines)
    stemmed = matrix_with_local_vocab(stemmed_lines)

    print(
        f'\n=== Tokenization ablation for "{document_names[target_play]}" '
        "(term-document + cosine; baseline = assignment tokenization) ==="
    )
    print(
        f"Stemmed local vocabulary size: {stemmed.shape[0]} "
        f"(assignment vocab size: {len(vocab)})."
    )
    variants = (
        ("baseline", baseline),
        ("no stop words", stopped),
        ("crude stemming (local vocab)", stemmed),
    )
    neighbor_sets = {}
    for label, matrix in variants:
        ranked = rank_plays(target_play, matrix, compute_cosine_similarity)
        neighbors = _top_neighbors(ranked, target_play, limit=10)
        neighbor_sets[label] = neighbors
        print(f"\n{label}:")
        for rank, play_id in enumerate(neighbors, start=1):
            print(f"  {rank}: {document_names[play_id]}")

    base = set(neighbor_sets["baseline"])
    for label in ("no stop words", "crude stemming (local vocab)"):
        overlap = len(base & set(neighbor_sets[label]))
        print(
            f"Top-10 overlap with baseline ({label}): {overlap}/10 "
            "(change ≠ improvement without external labels)."
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Optional Assignment 3 experiments.")
    parser.add_argument("--word", default="juliet", help="Word explorer query.")
    parser.add_argument("--play-index", type=int, default=0, help="Play index for rankings.")
    parser.add_argument("--window", type=int, default=2, help="Context window.")
    parser.add_argument("--characters", action="store_true", help="Run character comparison.")
    parser.add_argument("--limit-chars", type=int, default=25, help="Speakers to keep.")
    parser.add_argument("--ablation", action="store_true", help="Run stop-word/stem ablation.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run word explorer, play TF-IDF comparison, characters, and ablation.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_all = args.all

    lines, document_names, vocab = read_in_shakespeare()
    if not 0 <= args.play_index < len(document_names):
        raise SystemExit("play-index out of range for play_names.txt")

    print("Building term-document and TF-IDF matrices...")
    td_matrix = create_term_document_matrix(lines, document_names, vocab)
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    print("Building sparse term-context and PPMI matrices for word explorer...")
    tc_matrix = create_term_context_matrix_sparse(
        lines, vocab, context_window_size=args.window
    )
    ppmi_matrix = create_PPMI_matrix_sparse(tc_matrix)

    explore_word_neighbors(args.word, vocab, td_matrix, tc_matrix, ppmi_matrix)
    compare_play_rankings_raw_vs_tfidf(
        document_names, td_matrix, tf_idf_matrix, args.play_index
    )

    if args.characters or run_all:
        compare_characters(vocab, limit_chars=args.limit_chars, window=args.window)

    if args.ablation or run_all:
        stopword_stem_ablation(lines, document_names, vocab, args.play_index)

    print(
        "\nDone.  These experiments are optional; required assignment functions "
        "remain in main.py and are unchanged by this script."
    )
