"""Shakespeare play and word similarity using distributional semantics.

This file fills in the Assignment 3 starter stubs.  Function names, arguments,
and return shapes match the starter so the required interface stays compatible:
required matrix builders return NumPy arrays.

Optional helpers (sparse matrices, Clifford illustration) live below the
required functions and are used only when requested from ``__main__``.

Formulas follow CECS 429/529 Week 10 (log TF, IDF, set Jaccard, cosine) and
the assignment / textbook treatment of PPMI.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np

# SciPy is optional.  The required NumPy path works without it; the sparse path
# needs it because a dense |V|x|V| float32 matrix is about 2 GB of RAM.
try:
    from scipy import sparse as sp
except ImportError:  # pragma: no cover - exercised only when SciPy is absent
    sp = None


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _input_path(filename):
    """Find an input file beside this script, then in the current directory."""
    beside_script = Path(__file__).resolve().parent / filename
    if beside_script.exists():
        return beside_script
    return Path.cwd() / filename


def read_in_shakespeare():
    """Read the corpus, its vocabulary, and its ordered list of plays.

    Each returned line is ``(play_name, tokens)``.  Punctuation is replaced
    with spaces to match the starter's tokenization; tokens are lowercase.
    """
    lines = []
    with _input_path("will_play_text.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        reader = csv.reader(source, delimiter=";")
        for row_number, row in enumerate(reader, start=1):
            # Columns: line id, play, act/scene metadata, speaker, ..., text.
            if len(row) <= 5:
                raise ValueError(f"CSV row {row_number} has fewer than 6 columns")
            play_name = row[1]
            text = row[5]
            tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower().split()
            lines.append((play_name, tokens))

    with _input_path("vocab.txt").open(encoding="utf-8-sig") as source:
        vocab = [line.strip() for line in source if line.strip()]
    with _input_path("play_names.txt").open(encoding="utf-8-sig") as source:
        document_names = [line.strip() for line in source if line.strip()]

    return lines, document_names, vocab


def get_row_vector(matrix, row_id):
    """Return one row as a 1-D NumPy vector (works for dense or SciPy sparse)."""
    row = matrix[row_id, :]
    if sp is not None and sp.issparse(row):
        return np.asarray(row.toarray(), dtype=np.float64).reshape(-1)
    return np.asarray(row, dtype=np.float64).reshape(-1)


def get_column_vector(matrix, col_id):
    """Return one column as a 1-D NumPy vector (works for dense or SciPy sparse)."""
    column = matrix[:, col_id]
    if sp is not None and sp.issparse(column):
        return np.asarray(column.toarray(), dtype=np.float64).reshape(-1)
    return np.asarray(column, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# Required matrix construction (NumPy arrays, matching the starter)
# ---------------------------------------------------------------------------

def create_term_document_matrix(line_tuples, document_names, vocab):
    """Build a word-by-play matrix of raw token counts.

    Rows are vocabulary terms and columns are plays.  ``A[i, j]`` is how many
    times word ``i`` occurs in play ``j``.  Tokens missing from ``vocab`` are
    ignored.  Float32 keeps memory modest for the |V| x |plays| shape.
    """
    word_to_id = {word: i for i, word in enumerate(vocab)}
    play_to_id = {name: i for i, name in enumerate(document_names)}
    matrix = np.zeros((len(vocab), len(document_names)), dtype=np.float32)

    for play_name, tokens in line_tuples:
        doc_id = play_to_id.get(play_name)
        if doc_id is None:
            continue
        for token in tokens:
            word_id = word_to_id.get(token)
            if word_id is not None:
                matrix[word_id, doc_id] += 1.0
    return matrix


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    """Count nearby word pairs without letting context cross line boundaries.

    ``matrix[i, j]`` counts how often vocabulary word ``j`` appears within
    ``context_window_size`` positions to the left or right of word ``i``.
    Left and right neighbors are pooled together (undirected window), so the
    matrix does not encode order by itself.

    Returns a dense NumPy array, as the starter docstring requires.  For the
    full Shakespeare vocabulary (~22k types) this needs about 2 GB of RAM for
    float32; see ``create_term_context_matrix_sparse`` for an optional path.
    """
    if context_window_size < 0:
        raise ValueError("context_window_size must be non-negative")

    word_to_id = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float32)

    for _play_name, tokens in line_tuples:
        # Map once per line; -1 marks a token outside the provided vocabulary.
        ids = [word_to_id.get(token, -1) for token in tokens]
        for position, focus_id in enumerate(ids):
            if focus_id < 0:
                continue
            start = max(0, position - context_window_size)
            stop = min(len(ids), position + context_window_size + 1)
            for context_position in range(start, stop):
                if context_position == position:
                    continue  # A word is not its own context at distance zero.
                context_id = ids[context_position]
                if context_id >= 0:
                    matrix[focus_id, context_id] += 1.0
    return matrix


def create_PPMI_matrix(term_context_matrix):
    """Convert co-occurrence counts to positive pointwise mutual information.

    With joint and marginal probabilities estimated from the count matrix F:

        P(i, j) = F[i, j] / sum(F)
        P(i)    = sum_j F[i, j] / sum(F)
        P(j)    = sum_i F[i, j] / sum(F)
        PMI(i, j) = log2( P(i, j) / (P(i) P(j)) )
                  = log2( F[i, j] * sum(F) / (row_sum[i] * col_sum[j]) )

    PPMI keeps only positive associations: ``max(PMI, 0)``.  Zero-count pairs
    stay zero (we never take log of zero).  Rows are processed one at a time so
    we do not allocate a second full |V|x|V| temporary for the outer product.

    Returns a dense NumPy array to match the starter interface.
    """
    counts = np.asarray(term_context_matrix, dtype=np.float32)
    if counts.ndim != 2:
        raise ValueError("term_context_matrix must be two-dimensional")
    if np.any(counts < 0):
        raise ValueError("term_context_matrix cannot contain negative counts")

    total = float(counts.sum())
    result = np.zeros(counts.shape, dtype=np.float32)
    if total == 0:
        return result

    row_totals = counts.sum(axis=1)
    col_totals = counts.sum(axis=0)

    for row_id, row_total in enumerate(row_totals):
        if row_total == 0:
            continue
        observed = counts[row_id] > 0
        # Independence expectation for this row: P(i)P(j)*N = row_total*col_totals/N.
        expected = (row_total * col_totals) / total
        valid = observed & (expected > 0)
        result[row_id, valid] = np.maximum(
            np.log2(counts[row_id, valid] / expected[valid]), 0.0
        )
    return result


def create_tf_idf_matrix(term_document_matrix):
    """Apply logarithmic term frequency times inverse document frequency.

    Week 10 log-frequency convention (base 10):

        tf_weight(t, d) = 0                     if count = 0
                        = 1 + log10(count)      if count > 0

    so the lecture examples hold: 0→0, 1→1, 2→1.3, 10→2, 1000→4.

    Inverse document frequency (same lecture / Manning IR convention):

        idf(t) = log10( N / df(t) )

    where N is the number of plays and df(t) is how many plays contain t.
    A term in every play has idf 0.  Final entry: tf_weight * idf.

    Note: the assignment task list sometimes writes ``compute_tf_idf_matrix``;
    the starter defines ``create_tf_idf_matrix``, which we keep.
    """
    counts = np.asarray(term_document_matrix, dtype=np.float32)
    if counts.ndim != 2:
        raise ValueError("term_document_matrix must be two-dimensional")
    if np.any(counts < 0):
        raise ValueError("term_document_matrix cannot contain negative counts")

    number_of_documents = counts.shape[1]
    document_frequency = np.count_nonzero(counts, axis=1).astype(np.float32)
    idf = np.zeros(document_frequency.shape, dtype=np.float32)
    present = document_frequency > 0
    if number_of_documents > 0:
        idf[present] = np.log10(number_of_documents / document_frequency[present])

    tf = np.zeros_like(counts)
    positive = counts > 0
    tf[positive] = 1.0 + np.log10(counts[positive])
    return tf * idf[:, np.newaxis]


# Alias listed in the assignment PDF task list; same function as the starter name.
compute_tf_idf_matrix = create_tf_idf_matrix


# ---------------------------------------------------------------------------
# Similarity measures
# ---------------------------------------------------------------------------

def _as_vector(vector):
    """Flatten row/column vectors and reject non-finite values."""
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("similarity vectors must contain only finite values")
    return values


def compute_cosine_similarity(vector1, vector2):
    """Cosine of the angle between two vectors.

    cosine(v, w) = (v · w) / (|v| |w|).  If either vector has length 0, the
    angle is undefined; we define the similarity as 0.0 so ranking stays safe.
    """
    first, second = _as_vector(vector1), _as_vector(vector2)
    if first.shape != second.shape:
        raise ValueError("similarity vectors must have the same length")
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def compute_jaccard_similarity(vector1, vector2):
    """Set Jaccard overlap using nonzero coordinates as set membership.

    Week 10 defines Jaccard on sets: |A ∩ B| / |A ∪ B|.  Treating nonzero
    vector entries as "present" matches that lecture (and the assignment's
    Dice = 2J/(J+1) identity for sets).  Empty union → 0.0.
    """
    first, second = _as_vector(vector1), _as_vector(vector2)
    if first.shape != second.shape:
        raise ValueError("similarity vectors must have the same length")
    first_present, second_present = first != 0, second != 0
    union_size = np.count_nonzero(first_present | second_present)
    if union_size == 0:
        return 0.0
    intersection_size = np.count_nonzero(first_present & second_present)
    return float(intersection_size / union_size)


def compute_dice_similarity(vector1, vector2):
    """Dice coefficient on nonzero coordinates as sets.

    Dice = 2|A ∩ B| / (|A| + |B|).  For sets this equals (2 * J) / (J + 1)
    when J is the Jaccard index, which is the relationship stated in the
    assignment.  Empty sets → 0.0.
    """
    first, second = _as_vector(vector1), _as_vector(vector2)
    if first.shape != second.shape:
        raise ValueError("similarity vectors must have the same length")
    first_present, second_present = first != 0, second != 0
    denominator = np.count_nonzero(first_present) + np.count_nonzero(second_present)
    if denominator == 0:
        return 0.0
    intersection_size = np.count_nonzero(first_present & second_present)
    return float(2.0 * intersection_size / denominator)


# ---------------------------------------------------------------------------
# Ranking (includes the target; callers exclude it when printing top-k)
# ---------------------------------------------------------------------------

def _cosine_scores_against_columns(matrix, target_index):
    """Vectorized cosine of one column against every column."""
    if sp is not None and sp.issparse(matrix):
        dense_cols = matrix.tocsc()
        target = get_column_vector(dense_cols, target_index)
        target_norm = float(np.linalg.norm(target))
        # Column norms via squared entries; sparse-friendly.
        sq = dense_cols.power(2)
        col_norms = np.sqrt(_dense1d(sq.sum(axis=0)))
        dots = _dense1d(target @ dense_cols)
    else:
        data = np.asarray(matrix, dtype=np.float64)
        target = data[:, target_index]
        target_norm = float(np.linalg.norm(target))
        col_norms = np.linalg.norm(data, axis=0)
        dots = _dense1d(target @ data)

    scores = np.zeros(dots.shape[0], dtype=np.float64)
    if target_norm == 0:
        return scores
    denom = col_norms * target_norm
    nonzero = denom > 0
    scores[nonzero] = dots[nonzero] / denom[nonzero]
    return scores


def _cosine_scores_against_rows(matrix, target_index):
    """Vectorized cosine of one row against every row."""
    if sp is not None and sp.issparse(matrix):
        csr = matrix.tocsr()
        target = get_row_vector(csr, target_index)
        target_norm = float(np.linalg.norm(target))
        sq = csr.power(2)
        row_norms = np.sqrt(_dense1d(sq.sum(axis=1)))
        dots = _dense1d(csr @ target)
    else:
        data = np.asarray(matrix, dtype=np.float64)
        target = data[target_index]
        target_norm = float(np.linalg.norm(target))
        row_norms = np.linalg.norm(data, axis=1)
        dots = _dense1d(data @ target)

    scores = np.zeros(dots.shape[0], dtype=np.float64)
    if target_norm == 0:
        return scores
    denom = row_norms * target_norm
    nonzero = denom > 0
    scores[nonzero] = dots[nonzero] / denom[nonzero]
    return scores


def _sorted_indices_by_score(scores):
    """Sort indices by score descending; ties keep the lower original index."""
    order = np.argsort(-scores, kind="mergesort")
    return order.astype(int).tolist()


def _dense1d(values):
    """Convert dense or SciPy sparse vectors/matrices to a 1-D float64 array."""
    if sp is not None and sp.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _is_cosine_fn(similarity_fn):
    return getattr(similarity_fn, "__name__", "") == "compute_cosine_similarity"


def _is_jaccard_fn(similarity_fn):
    return getattr(similarity_fn, "__name__", "") == "compute_jaccard_similarity"


def _is_dice_fn(similarity_fn):
    return getattr(similarity_fn, "__name__", "") == "compute_dice_similarity"


def _set_overlap_scores_against_rows(matrix, target_index, kind):
    """Vectorized set Jaccard or Dice of one row against every row.

    Nonzero entries define set membership (same convention as the scalar
    similarity helpers).  Intersection sizes are a sparse/dense matrix-vector
    product on a 0/1 matrix, which beginners can read as "shared on-dimensions."
    """
    if kind not in ("jaccard", "dice"):
        raise ValueError("kind must be 'jaccard' or 'dice'")

    if sp is not None and sp.issparse(matrix):
        binary = (matrix != 0).astype(np.float64).tocsr()
        target = binary.getrow(target_index).transpose()
        intersections = _dense1d(binary @ target)
        row_sizes = _dense1d(binary.sum(axis=1))
    else:
        data = np.asarray(matrix, dtype=np.float64)
        binary = (data != 0).astype(np.float64)
        target = binary[target_index]
        intersections = _dense1d(binary @ target)
        row_sizes = _dense1d(binary.sum(axis=1))

    target_size = float(row_sizes[target_index])
    if kind == "jaccard":
        unions = row_sizes + target_size - intersections
        scores = np.zeros(intersections.shape[0], dtype=np.float64)
        nonzero = unions > 0
        scores[nonzero] = intersections[nonzero] / unions[nonzero]
        return scores

    denominators = row_sizes + target_size
    scores = np.zeros(intersections.shape[0], dtype=np.float64)
    nonzero = denominators > 0
    scores[nonzero] = 2.0 * intersections[nonzero] / denominators[nonzero]
    return scores


def _set_overlap_scores_against_columns(matrix, target_index, kind):
    """Same set overlap idea as rows, but comparing document columns."""
    if kind not in ("jaccard", "dice"):
        raise ValueError("kind must be 'jaccard' or 'dice'")

    if sp is not None and sp.issparse(matrix):
        binary = (matrix != 0).astype(np.float64).tocsc()
        target = binary.getcol(target_index)
        intersections = _dense1d(target.transpose() @ binary)
        col_sizes = _dense1d(binary.sum(axis=0))
    else:
        data = np.asarray(matrix, dtype=np.float64)
        binary = (data != 0).astype(np.float64)
        target = binary[:, target_index]
        intersections = _dense1d(target @ binary)
        col_sizes = _dense1d(binary.sum(axis=0))

    target_size = float(col_sizes[target_index])
    if kind == "jaccard":
        unions = col_sizes + target_size - intersections
        scores = np.zeros(intersections.shape[0], dtype=np.float64)
        nonzero = unions > 0
        scores[nonzero] = intersections[nonzero] / unions[nonzero]
        return scores

    denominators = col_sizes + target_size
    scores = np.zeros(intersections.shape[0], dtype=np.float64)
    nonzero = denominators > 0
    scores[nonzero] = 2.0 * intersections[nonzero] / denominators[nonzero]
    return scores


def rank_plays(target_play_index, term_document_matrix, similarity_fn):
    """Rank all plays by similarity to the selected play (including itself).

    Returns a length-n list of play indices in decreasing similarity order.
    When printing a "top 10 nearest" list for the writeup, skip the target
    index explicitly so self-similarity is not reported as a neighbor.
    Tied scores keep the earlier play index (stable sort).
    """
    if sp is not None and sp.issparse(term_document_matrix):
        number_of_plays = term_document_matrix.shape[1]
    else:
        matrix = np.asarray(term_document_matrix)
        if matrix.ndim != 2:
            raise ValueError("term_document_matrix must be two-dimensional")
        number_of_plays = matrix.shape[1]
        term_document_matrix = matrix

    if not 0 <= target_play_index < number_of_plays:
        raise IndexError("target_play_index is outside the matrix")

    if _is_cosine_fn(similarity_fn):
        scores = _cosine_scores_against_columns(
            term_document_matrix, target_play_index
        )
        return _sorted_indices_by_score(scores)
    if _is_jaccard_fn(similarity_fn):
        scores = _set_overlap_scores_against_columns(
            term_document_matrix, target_play_index, "jaccard"
        )
        return _sorted_indices_by_score(scores)
    if _is_dice_fn(similarity_fn):
        scores = _set_overlap_scores_against_columns(
            term_document_matrix, target_play_index, "dice"
        )
        return _sorted_indices_by_score(scores)

    target = get_column_vector(term_document_matrix, target_play_index)
    scores = [
        similarity_fn(target, get_column_vector(term_document_matrix, play_id))
        for play_id in range(number_of_plays)
    ]
    # Python's stable sort preserves original play order for equal scores.
    return sorted(range(number_of_plays), key=lambda i: scores[i], reverse=True)


def rank_words(target_word_index, matrix, similarity_fn):
    """Rank all vocabulary words by similarity between their matrix rows.

    Returns every word index, including the target.  Exclude the target when
    displaying nearest neighbors.  Cosine / Jaccard / Dice use matrix-level
    fast paths that match the scalar helpers' definitions.
    """
    if sp is not None and sp.issparse(matrix):
        number_of_words = matrix.shape[0]
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ValueError("word matrix must be two-dimensional")
        number_of_words = dense.shape[0]
        matrix = dense

    if not 0 <= target_word_index < number_of_words:
        raise IndexError("target_word_index is outside the matrix")

    if _is_cosine_fn(similarity_fn):
        scores = _cosine_scores_against_rows(matrix, target_word_index)
        return _sorted_indices_by_score(scores)
    if _is_jaccard_fn(similarity_fn):
        scores = _set_overlap_scores_against_rows(matrix, target_word_index, "jaccard")
        return _sorted_indices_by_score(scores)
    if _is_dice_fn(similarity_fn):
        scores = _set_overlap_scores_against_rows(matrix, target_word_index, "dice")
        return _sorted_indices_by_score(scores)

    target = get_row_vector(matrix, target_word_index)
    scores = [
        similarity_fn(target, get_row_vector(matrix, word_id))
        for word_id in range(number_of_words)
    ]
    return sorted(range(number_of_words), key=lambda i: scores[i], reverse=True)


# ---------------------------------------------------------------------------
# Optional sparse path (SciPy).  Not part of the starter return-type contract.
# ---------------------------------------------------------------------------

def _require_scipy():
    if sp is None:
        raise ImportError(
            "SciPy is required for the sparse matrix path. "
            "Install it with: pip install scipy"
        )


def create_term_context_matrix_sparse(line_tuples, vocab, context_window_size=1):
    """Same counts as ``create_term_context_matrix``, stored as CSR float32.

    Tradeoff versus the starter's dense NumPy array:
    - Dense matches the starter docstring exactly but needs ~2 GB for |V|≈22k.
    - Sparse uses far less memory when most co-occurrence cells are zero, but
      the return type is ``scipy.sparse.csr_matrix``, not ``numpy.ndarray``.
    Keep the dense function for interface compatibility; use this helper when
    running on the full corpus.
    """
    from collections import Counter

    _require_scipy()
    if context_window_size < 0:
        raise ValueError("context_window_size must be non-negative")

    word_to_id = {word: i for i, word in enumerate(vocab)}
    pair_counts = Counter()

    for _play_name, tokens in line_tuples:
        ids = [word_to_id.get(token, -1) for token in tokens]
        for position, focus_id in enumerate(ids):
            if focus_id < 0:
                continue
            start = max(0, position - context_window_size)
            stop = min(len(ids), position + context_window_size + 1)
            for context_position in range(start, stop):
                if context_position == position:
                    continue
                context_id = ids[context_position]
                if context_id >= 0:
                    pair_counts[(focus_id, context_id)] += 1

    n = len(vocab)
    if not pair_counts:
        return sp.csr_matrix((n, n), dtype=np.float32)
    rows, cols, data = zip(
        *((i, j, value) for (i, j), value in pair_counts.items())
    )
    matrix = sp.coo_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    return matrix.tocsr()


def create_PPMI_matrix_sparse(term_context_matrix):
    """PPMI over a sparse (or dense) co-occurrence matrix; returns CSR.

    Only nonzero count cells can receive nonzero PPMI, so the result stays
    sparse.  Same PMI formula as ``create_PPMI_matrix``.
    """
    _require_scipy()
    if sp.issparse(term_context_matrix):
        counts = term_context_matrix.tocsr().astype(np.float64)
    else:
        counts = sp.csr_matrix(np.asarray(term_context_matrix, dtype=np.float64))

    total = float(counts.sum())
    if total == 0:
        return sp.csr_matrix(counts.shape, dtype=np.float32)

    row_totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    col_totals = np.asarray(counts.sum(axis=0)).reshape(-1)

    coo = counts.tocoo()
    expected = (row_totals[coo.row] * col_totals[coo.col]) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(coo.data / expected)
    pmi = np.maximum(pmi, 0.0)
    pmi[~np.isfinite(pmi)] = 0.0

    result = sp.coo_matrix((pmi.astype(np.float32), (coo.row, coo.col)), shape=coo.shape)
    result.eliminate_zeros()
    return result.tocsr()


# ---------------------------------------------------------------------------
# Optional Clifford / geometric-product illustration (inspired by clifford.pdf)
# ---------------------------------------------------------------------------

def _hand_worked_geometric_product_example():
    """Tiny 2-D check independent of the Shakespeare corpus.

    For a = (2, 0) and b = (1, 1):
      a · b = 2*1 + 0*1 = 2
      a ∧ b in 2-D is the scalar area a_x b_y - a_y b_x = 2*1 - 0*1 = 2
      so ab = 2 + 2 e12  (grade-0 + grade-2)
      ba has the same dot product and the negated wedge: 2 + (-2) e12.
    """
    a = np.array([2.0, 0.0])
    b = np.array([1.0, 1.0])
    dot_ab = float(np.dot(a, b))
    wedge_ab = float(a[0] * b[1] - a[1] * b[0])
    dot_ba = float(np.dot(b, a))
    wedge_ba = float(b[0] * a[1] - b[1] * a[0])
    return {
        "a": a,
        "b": b,
        "dot_ab": dot_ab,
        "wedge_ab": wedge_ab,
        "dot_ba": dot_ba,
        "wedge_ba": wedge_ba,
    }


def demonstrate_geometric_product(
    matrix,
    vocab,
    word_a="romeo",
    word_b="juliet",
    n_components=2,
):
    """Illustrate ab = a·b + a∧b on two reduced word vectors.

    This is a small teaching demo inspired by Pustejovsky's Functional Geometric
    Algebra paper (clifford.pdf).  It is NOT an implementation of that full
    framework: a Clifford algebra over the raw |V|-dimensional vectors would
    have size 2^|V|, which is impossible here.

    Reduction: keep the ``n_components`` dimensions (2 or 3) where
    |a_i| + |b_i| is largest.  That preserves the coordinates that contribute
    most to both vectors' magnitudes, but it discards the rest of the space,
    so numerical values are only meaningful relative to this projection.

    The context / PPMI matrices count left and right neighbors together, so a
    nonzero wedge after reduction does NOT prove the model encodes word order
    or character relationships — only that the projected vectors are not
    parallel in the chosen subspace.
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for this beginner demo")

    word_to_id = {word: i for i, word in enumerate(vocab)}
    for label, word in (("word_a", word_a), ("word_b", word_b)):
        if word not in word_to_id:
            raise KeyError(f"{label} {word!r} is not in the vocabulary")

    full_a = get_row_vector(matrix, word_to_id[word_a])
    full_b = get_row_vector(matrix, word_to_id[word_b])
    if np.allclose(full_a, 0) or np.allclose(full_b, 0):
        raise ValueError(
            "Refusing geometric product: at least one selected word has a "
            "zero vector in this matrix (no observed co-occurrence signal)."
        )

    importance = np.abs(full_a) + np.abs(full_b)
    # mergesort keeps lower indices first on ties for reproducibility.
    chosen = np.argsort(-importance, kind="mergesort")[:n_components]
    a = full_a[chosen]
    b = full_b[chosen]

    # Grade-0 part: ordinary dot product (symmetric).
    dot_ab = float(np.dot(a, b))
    dot_ba = float(np.dot(b, a))

    # Grade-2 part: oriented parallelogram.  In 2-D the bivector has one
    # coefficient (e12).  In 3-D we report the three coefficients (e12, e13, e23).
    if n_components == 2:
        wedge_ab = np.array([a[0] * b[1] - a[1] * b[0]], dtype=np.float64)
        wedge_ba = np.array([b[0] * a[1] - b[1] * a[0]], dtype=np.float64)
        wedge_basis = ("e12",)
    else:
        wedge_ab = np.array(
            [
                a[0] * b[1] - a[1] * b[0],
                a[0] * b[2] - a[2] * b[0],
                a[1] * b[2] - a[2] * b[1],
            ],
            dtype=np.float64,
        )
        wedge_ba = -wedge_ab
        wedge_basis = ("e12", "e13", "e23")

    hand = _hand_worked_geometric_product_example()

    return {
        "word_a": word_a,
        "word_b": word_b,
        "chosen_dimensions": chosen.tolist(),
        "reduced_a": a,
        "reduced_b": b,
        "dot_ab": dot_ab,
        "dot_ba": dot_ba,
        "wedge_ab": wedge_ab,
        "wedge_ba": wedge_ba,
        "wedge_basis": wedge_basis,
        "hand_worked": hand,
        "note": (
            "Illustration only: geometric product on a reduced subspace, "
            "not full Functional Geometric Algebra over the vocabulary."
        ),
    }


def _print_geometric_product_demo(demo):
    """Readable console output for the optional Clifford illustration."""
    print("\n=== Optional Clifford / geometric-product demo ===")
    print(demo["note"])
    hand = demo["hand_worked"]
    print(
        "Hand-worked 2-D check (not corpus data): "
        f"a={hand['a']}, b={hand['b']}; "
        f"a·b={hand['dot_ab']}, a∧b={hand['wedge_ab']}; "
        f"b·a={hand['dot_ba']}, b∧a={hand['wedge_ba']}"
    )
    print(
        f"Corpus words: {demo['word_a']!r} and {demo['word_b']!r} "
        f"(PPMI/context rows projected onto dims {demo['chosen_dimensions']})"
    )
    print(f"Reduced {demo['word_a']}: {demo['reduced_a']}")
    print(f"Reduced {demo['word_b']}: {demo['reduced_b']}")
    print(f"Dot product a·b = {demo['dot_ab']:.6f} (b·a = {demo['dot_ba']:.6f})")
    wedge_parts = ", ".join(
        f"{name}={value:.6f}"
        for name, value in zip(demo["wedge_basis"], demo["wedge_ab"])
    )
    print(f"Wedge a∧b components: {wedge_parts}")
    wedge_rev = ", ".join(
        f"{name}={value:.6f}"
        for name, value in zip(demo["wedge_basis"], demo["wedge_ba"])
    )
    print(f"Wedge b∧a components: {wedge_rev}")
    print(
        "Observation: reversing order leaves the dot product unchanged and "
        "negates the wedge (antisymmetric)."
    )


# ---------------------------------------------------------------------------
# Display helpers used by the assignment-style __main__ block
# ---------------------------------------------------------------------------

def _top_neighbors(ranked_ids, target_id, limit=10):
    """Drop the target id, then take the first ``limit`` remaining ids."""
    return [item_id for item_id in ranked_ids if item_id != target_id][:limit]


def _print_play_ranking(document_names, target_play, ranked, similarity_name, representation):
    others = _top_neighbors(ranked, target_play, limit=10)
    print(
        f'\nThe 10 most similar plays to "{document_names[target_play]}" '
        f"using {similarity_name} on {representation} are:"
    )
    for rank, play_id in enumerate(others, start=1):
        print(f"{rank}: {document_names[play_id]}")


def _print_word_ranking(vocab, target_word, target_id, ranked, similarity_name, representation):
    others = _top_neighbors(ranked, target_id, limit=10)
    print(
        f'\nThe 10 most similar words to "{target_word}" using '
        f"{similarity_name} on {representation} are:"
    )
    for rank, word_id in enumerate(others, start=1):
        print(f"{rank}: {vocab[word_id]}")


def _closest_and_farthest_plays(document_names, td_matrix):
    """Pairwise cosine extremes, ignoring self-similarity."""
    play_pairs = []
    for first in range(len(document_names)):
        for second in range(first + 1, len(document_names)):
            score = compute_cosine_similarity(
                get_column_vector(td_matrix, first),
                get_column_vector(td_matrix, second),
            )
            play_pairs.append((score, first, second))
    if not play_pairs:
        return
    for label, pair in (("Closest", max(play_pairs)), ("Most distant", min(play_pairs))):
        score, first, second = pair
        print(
            f"{label} plays: {document_names[first]} / "
            f"{document_names[second]} (cosine {score:.4f})"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Shakespeare vector-space assignment runner."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the randomly chosen target play (default: 42).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Context window size for the term-context matrix (default: 2).",
    )
    parser.add_argument(
        "--dense-context",
        action="store_true",
        help=(
            "Build dense NumPy term-context / PPMI matrices (starter types). "
            "Needs ~2+ GB RAM for the full vocabulary."
        ),
    )
    parser.add_argument(
        "--clifford",
        action="store_true",
        help="After PPMI is built, run the optional geometric-product demo.",
    )
    parser.add_argument(
        "--skip-slow-set-sims",
        action="store_true",
        help=(
            "For word rankings on large matrices, run cosine only "
            "(Jaccard/Dice over 22k rows are much slower)."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    lines, document_names, vocab = read_in_shakespeare()
    print(
        f"Loaded {len(lines)} lines, {len(document_names)} plays, "
        f"{len(vocab)} vocabulary types."
    )

    print("Computing term-document matrix...")
    td_matrix = create_term_document_matrix(lines, document_names, vocab)

    print("Computing TF-IDF matrix...")
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)

    # Assignment: closest / most distant plays ignoring self-similarity.
    _closest_and_farthest_plays(document_names, td_matrix)

    target_play = random.randrange(len(document_names))
    print(f'\nTarget play (seed={args.seed}): "{document_names[target_play]}"')

    # Raw counts vs TF-IDF with cosine (useful for the written discussion).
    for label, representation in (("raw counts", td_matrix), ("TF-IDF", tf_idf_matrix)):
        order = rank_plays(target_play, representation, compute_cosine_similarity)
        _print_play_ranking(
            document_names, target_play, order, "compute_cosine_similarity", label
        )

    word_to_id = {word: i for i, word in enumerate(vocab)}
    target_word = "juliet"
    if target_word not in word_to_id:
        raise ValueError(f"{target_word!r} is not in vocab.txt")
    target_word_id = word_to_id[target_word]

    # Term-document word vectors are only |plays|-dimensional.
    order = rank_words(target_word_id, td_matrix, compute_cosine_similarity)
    _print_word_ranking(
        vocab,
        target_word,
        target_word_id,
        order,
        "compute_cosine_similarity",
        "term-document counts",
    )

    use_sparse = not args.dense_context
    if use_sparse:
        print(
            "\nComputing term-context matrix with the optional SciPy sparse path "
            f"(window={args.window})..."
        )
        print(
            "Note: create_term_context_matrix() still returns dense NumPy when "
            "called directly; sparse is used here to avoid a ~2 GB allocation."
        )
        tc_matrix = create_term_context_matrix_sparse(
            lines, vocab, context_window_size=args.window
        )
        print("Computing PPMI matrix (sparse)...")
        ppmi_matrix = create_PPMI_matrix_sparse(tc_matrix)
        nnz = tc_matrix.nnz
        density = nnz / (tc_matrix.shape[0] * tc_matrix.shape[1])
        print(
            f"Term-context nonzero entries: {nnz} "
            f"(density {density:.6f}; sparse saves memory vs dense)."
        )
    else:
        print(
            f"\nComputing dense term-context matrix (window={args.window})..."
        )
        tc_matrix = create_term_context_matrix(
            lines, vocab, context_window_size=args.window
        )
        print("Computing dense PPMI matrix...")
        ppmi_matrix = create_PPMI_matrix(tc_matrix)

    similarity_functions = [
        compute_cosine_similarity,
        compute_jaccard_similarity,
        compute_dice_similarity,
    ]

    for similarity_fn in similarity_functions:
        ranked = rank_plays(target_play, td_matrix, similarity_fn)
        _print_play_ranking(
            document_names,
            target_play,
            ranked,
            similarity_fn.__name__,
            "term-document counts",
        )

    word_sim_fns = similarity_functions
    if args.skip_slow_set_sims:
        word_sim_fns = [compute_cosine_similarity]
        print(
            "\n(--skip-slow-set-sims) Word rankings will use cosine only."
        )

    for label, word_matrix in (
        ("term-context frequency", tc_matrix),
        ("PPMI", ppmi_matrix),
    ):
        for similarity_fn in word_sim_fns:
            print(
                f"\nRanking words vs {target_word!r} with "
                f"{similarity_fn.__name__} on {label}..."
            )
            ranked = rank_words(target_word_id, word_matrix, similarity_fn)
            _print_word_ranking(
                vocab,
                target_word,
                target_word_id,
                ranked,
                similarity_fn.__name__,
                label,
            )

    if args.clifford:
        # Run only after the PPMI matrix exists, as requested.
        try:
            demo = demonstrate_geometric_product(
                ppmi_matrix, vocab, word_a="romeo", word_b="juliet", n_components=2
            )
            _print_geometric_product_demo(demo)
        except (KeyError, ValueError) as exc:
            print(f"\nClifford demo skipped: {exc}")
