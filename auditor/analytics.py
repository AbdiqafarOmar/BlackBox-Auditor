"""Documented metrics for comparing audit runs."""

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate explicit metrics for each provider/model pair.

    violation_rate = tagged successful responses / successful responses
    refusal_rate = detected refusals / successful responses
    error_rate = failed requests / attempted requests
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("provider", "unknown"), row.get("model", "unknown"))].append(row)
    summaries: List[Dict[str, Any]] = []
    for (provider, model), group in sorted(groups.items()):
        attempted = len(group)
        successful = [row for row in group if not row.get("error")]
        violations = [row for row in successful if row.get("violation_count", 0) > 0]
        refusals = [row for row in successful if row.get("refused")]
        latencies = [float(row.get("latency_ms", 0.0)) for row in successful]
        tag_counts = Counter(tag for row in successful for tag in row.get("violations", []))
        summaries.append({
            "provider": provider, "model": model,
            "attempted_probes": attempted, "successful_probes": len(successful),
            "violation_rate": _rate(len(violations), len(successful)),
            "refusal_rate": _rate(len(refusals), len(successful)),
            "error_rate": _rate(attempted - len(successful), attempted),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "unique_violation_tags": len(tag_counts),
            "most_common_violation": tag_counts.most_common(1)[0][0] if tag_counts else "None",
        })
    return summaries


def mutation_breakdown(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return violation rates by provider and mutation strategy index."""
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("provider", "unknown"), row.get("mutation_index", 0))].append(row)
    result = []
    for (provider, mutation_index), group in sorted(groups.items()):
        successful = [row for row in group if not row.get("error")]
        tagged = sum(row.get("violation_count", 0) > 0 for row in successful)
        result.append({
            "provider": provider, "mutation_index": mutation_index,
            "probe_count": len(group),
            "violation_rate": _rate(tagged, len(successful)),
        })
    return result


def cluster_outputs(rows: Iterable[Dict[str, Any]], max_clusters: int = 5) -> List[Dict[str, Any]]:
    """Cluster successful non-empty outputs and expose representative terms.

    TF-IDF keeps the public demo lightweight and reproducible; K-means uses a
    fixed random state and initialization count.
    """
    payload = [dict(row) for row in rows if not row.get("error") and row.get("output")]
    if len(payload) < 2:
        return []
    texts = [row["output"] for row in payload]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    matrix = vectorizer.fit_transform(texts)
    unique_texts = len(set(texts))
    n_clusters = min(max_clusters, max(1, int(len(texts) ** 0.5)), unique_texts)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    terms = vectorizer.get_feature_names_out()
    top_terms = {}
    for cluster_id, center in enumerate(model.cluster_centers_):
        indices = center.argsort()[-3:][::-1]
        top_terms[cluster_id] = ", ".join(terms[index] for index in indices)
    clustered = []
    for row, label in zip(payload, labels):
        clustered.append({
            "provider": row.get("provider", "unknown"),
            "base_target": row.get("base_target", ""),
            "mutation_index": row.get("mutation_index", 0),
            "cluster": int(label),
            "representative_terms": top_terms[int(label)],
            "output": row["output"],
        })
    return clustered
