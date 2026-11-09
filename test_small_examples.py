"""Small, hand-checkable tests for the required Assignment 3 functions.

Run with:

    python test_small_examples.py
"""

from __future__ import annotations

import numpy as np

from main import (
    compute_cosine_similarity,
    compute_dice_similarity,
    compute_jaccard_similarity,
    create_PPMI_matrix,
    create_PPMI_matrix_sparse,
    create_term_context_matrix,
    create_term_context_matrix_sparse,
    create_term_document_matrix,
    create_tf_idf_matrix,
    rank_plays,
    rank_words,
)


def assert_close(actual, expected, label, tol=1e-6):
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def test_term_document_and_tfidf():
    lines = [
        ("PlayA", ["battle", "battle", "fool"]),
        ("PlayB", ["battle", "crown"]),
        ("PlayC", ["fool", "fool", "fool"]),
    ]
    docs = ["PlayA", "PlayB", "PlayC"]
    vocab = ["battle", "fool", "crown", "absent"]
    td = create_term_document_matrix(lines, docs, vocab)

    expected = np.array(
        [
            [2, 1, 0],
            [1, 0, 3],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    if not np.array_equal(td, expected):
        raise AssertionError(f"term-document mismatch:\n{td}\n!=\n{expected}")

    # Lecture log-TF: count 2 → 1 + log10(2) ≈ 1.3010; count 1 → 1; count 0 → 0.
    tf_idf = create_tf_idf_matrix(td)
    # battle appears in 2/3 plays → idf = log10(3/2)
    idf_battle = np.log10(3 / 2)
    assert_close(tf_idf[0, 0], (1 + np.log10(2)) * idf_battle, "tf-idf battle/PlayA")
    assert_close(tf_idf[0, 1], (1 + np.log10(1)) * idf_battle, "tf-idf battle/PlayB")
    assert_close(tf_idf[0, 2], 0.0, "tf-idf battle/PlayC")
    # absent never occurs → idf left 0, row stays 0
    if not np.allclose(tf_idf[3], 0):
        raise AssertionError("absent term should stay zero under tf-idf")
    print("PASS term-document + tf-idf")


def test_term_context_and_ppmi():
    # One line: a b a  with window 1.
    # Positions: 0:a contexts {b}; 1:b contexts {a,a}; 2:a contexts {b}.
    lines = [("P", ["a", "b", "a"])]
    vocab = ["a", "b", "c"]
    tc = create_term_context_matrix(lines, vocab, context_window_size=1)
    expected = np.array(
        [
            [0, 2, 0],  # each 'a' sees one 'b'
            [2, 0, 0],  # 'b' sees two 'a's
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    if not np.array_equal(tc, expected):
        raise AssertionError(f"term-context mismatch:\n{tc}")

    ppmi = create_PPMI_matrix(tc)
    # a and b always co-occur relative to chance → positive PPMI; diagonals 0.
    if ppmi[0, 1] <= 0 or ppmi[1, 0] <= 0:
        raise AssertionError(f"expected positive PPMI for (a,b), got {ppmi}")
    if ppmi[0, 0] != 0 or ppmi[1, 1] != 0:
        raise AssertionError("zero co-occurrence cells must stay zero in PPMI")
    if not np.allclose(ppmi[2], 0) or not np.allclose(ppmi[:, 2], 0):
        raise AssertionError("unseen word c must stay zero")

    sparse_tc = create_term_context_matrix_sparse(lines, vocab, context_window_size=1)
    if not np.allclose(sparse_tc.toarray(), expected):
        raise AssertionError("sparse term-context disagrees with dense")
    sparse_ppmi = create_PPMI_matrix_sparse(sparse_tc)
    if not np.allclose(sparse_ppmi.toarray(), ppmi, atol=1e-5):
        raise AssertionError("sparse PPMI disagrees with dense")
    print("PASS term-context + PPMI (dense and sparse)")


def test_similarities_and_ranking():
    v = np.array([1.0, 0.0, 1.0])
    w = np.array([1.0, 1.0, 0.0])
    z = np.zeros(3)

    assert_close(compute_cosine_similarity(v, v), 1.0, "cosine identical")
    assert_close(compute_cosine_similarity(v, z), 0.0, "cosine with zero")
    # Jaccard on supports {0,2} and {0,1}: intersection 1, union 3 → 1/3
    assert_close(compute_jaccard_similarity(v, w), 1.0 / 3.0, "jaccard")
    # Dice = 2*1 / (2+2) = 0.5; also (2J)/(J+1) = (2/3)/(4/3) = 0.5
    assert_close(compute_dice_similarity(v, w), 0.5, "dice")
    j = compute_jaccard_similarity(v, w)
    assert_close(
        compute_dice_similarity(v, w),
        (2 * j) / (j + 1),
        "dice from jaccard identity",
    )

    # Columns: play0 close to play1; play2 orthogonal-ish / empty-ish.
    td = np.array(
        [
            [5.0, 4.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 7.0],
        ]
    )
    ranked = rank_plays(0, td, compute_cosine_similarity)
    if ranked[0] != 0:
        raise AssertionError("target play should rank first (self cosine 1)")
    if ranked[1] != 1:
        raise AssertionError("play1 should be nearest neighbor of play0")

    # Rows for words.
    word_matrix = td  # reuse
    word_ranked = rank_words(0, word_matrix, compute_cosine_similarity)
    if word_ranked[0] != 0:
        raise AssertionError("target word should rank first")
    print("PASS similarities + ranking")



if __name__ == "__main__":
    test_term_document_and_tfidf()
    test_term_context_and_ppmi()
    test_similarities_and_ranking()
    print("\nAll small examples passed.")
