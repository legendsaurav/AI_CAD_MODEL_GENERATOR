"""
memory/retrieval.py — Production Memory Retrieval System
==========================================================
Retrieves construction patterns from the pattern database using
both rule-based keyword matching and vector similarity search.

Retrieval strategies:
  1. Keyword matching (fast, rule-based fallback)
  2. Cosine similarity on TF-IDF feature vectors (semantic search)
  3. Composite scoring with recency weighting

Integrates with shared-schemas RetrievedMemory for PlanningTrace.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from memory.pattern_database import PatternDatabase

logger = logging.getLogger("cad_planner.memory.retrieval")


class RetrievedMemory:
    """A memory retrieval result with confidence and provenance."""

    def __init__(
        self,
        pattern_name: str,
        construction_sequence: List[str],
        relevance_score: float,
        match_method: str,
    ) -> None:
        self.pattern_name = pattern_name
        self.construction_sequence = construction_sequence
        self.relevance_score = relevance_score
        self.match_method = match_method

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "construction_sequence": self.construction_sequence,
            "relevance_score": round(self.relevance_score, 4),
            "match_method": self.match_method,
        }


class MemoryRetrieval:
    """
    Multi-strategy construction pattern retrieval.

    Combines keyword-based and similarity-based retrieval,
    ranking results by composite relevance score.
    """

    # Semantic keyword associations for rule-based matching
    _KEYWORD_MAP: Dict[str, List[str]] = {
        "bearing_seat": ["bearing", "seat", "journal", "bushing", "bore"],
        "counterbore_hole": ["counterbore", "cbore", "socket", "cap_screw"],
        "flange": ["flange", "collar", "lip", "rim", "bolt_circle"],
        "standard_hole": ["hole", "bore", "drill", "through_hole", "blind_hole"],
        "keyway": ["keyway", "key_slot", "key_seat", "spline"],
        "chamfer_edge": ["chamfer", "bevel", "edge_break", "deburr"],
        "fillet_edge": ["fillet", "radius", "round", "blend"],
        "pocket": ["pocket", "recess", "cavity", "slot"],
        "boss": ["boss", "pad", "protrusion", "post", "standoff"],
        "rib": ["rib", "web", "stiffener", "gusset"],
        "shell": ["shell", "hollow", "thin_wall", "enclosure"],
        "thread": ["thread", "tap", "screw", "bolt"],
    }

    def __init__(self, db: PatternDatabase) -> None:
        self.db = db
        # Build TF-IDF vocabulary from keywords
        self._vocab = self._build_vocabulary()

    def retrieve(
        self,
        feature_semantics: str,
        top_k: int = 3,
    ) -> Optional[List[str]]:
        """
        Backward-compatible API: return best matching pattern sequence.

        Args:
            feature_semantics: Semantic description of the feature.

        Returns:
            Construction sequence list, or None if no match found.
        """
        results = self.retrieve_ranked(feature_semantics, top_k=1)
        if results:
            return results[0].construction_sequence
        return None

    def retrieve_ranked(
        self,
        feature_semantics: str,
        top_k: int = 3,
        min_relevance: float = 0.1,
    ) -> List[RetrievedMemory]:
        """
        Retrieve and rank patterns by composite relevance score.

        Args:
            feature_semantics: Semantic description of the feature.
            top_k: Maximum number of results to return.
            min_relevance: Minimum relevance threshold.

        Returns:
            List of RetrievedMemory sorted by relevance (descending).
        """
        candidates: List[RetrievedMemory] = []
        feature_lower = feature_semantics.lower()
        tokens = set(feature_lower.replace("_", " ").split())

        # Strategy 1: Keyword matching
        for pattern_name, keywords in self._KEYWORD_MAP.items():
            keyword_score = self._keyword_match_score(tokens, keywords, feature_lower)
            if keyword_score > 0:
                sequence = self.db.get_pattern(pattern_name)
                if sequence:
                    candidates.append(RetrievedMemory(
                        pattern_name=pattern_name,
                        construction_sequence=sequence,
                        relevance_score=keyword_score,
                        match_method="keyword",
                    ))

        # Strategy 2: TF-IDF similarity
        query_vec = self._tfidf_vector(tokens)
        for pattern_name in self.db.get_all_patterns():
            pattern_keywords = self._KEYWORD_MAP.get(pattern_name, [])
            pattern_tokens = set()
            for kw in pattern_keywords:
                pattern_tokens.update(kw.replace("_", " ").split())

            pattern_vec = self._tfidf_vector(pattern_tokens)
            sim = self._cosine_similarity(query_vec, pattern_vec)

            if sim > min_relevance:
                sequence = self.db.get_pattern(pattern_name)
                if sequence:
                    # Check if already added by keyword matching
                    existing = next(
                        (c for c in candidates if c.pattern_name == pattern_name),
                        None,
                    )
                    if existing:
                        # Boost score with similarity
                        existing.relevance_score = max(
                            existing.relevance_score, sim
                        )
                        existing.match_method = "keyword+tfidf"
                    else:
                        candidates.append(RetrievedMemory(
                            pattern_name=pattern_name,
                            construction_sequence=sequence,
                            relevance_score=sim,
                            match_method="tfidf",
                        ))

        # Sort by relevance descending
        candidates.sort(key=lambda c: c.relevance_score, reverse=True)

        # Filter and truncate
        results = [c for c in candidates if c.relevance_score >= min_relevance]
        results = results[:top_k]

        logger.info(
            "Memory retrieval for '%s': %d candidates, returning %d",
            feature_semantics, len(candidates), len(results),
        )

        return results

    # ------------------------------------------------------------------
    # Keyword matching
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_match_score(
        query_tokens: set, keywords: List[str], full_query: str
    ) -> float:
        """Score by keyword overlap, with bonus for exact substring match."""
        if not keywords:
            return 0.0

        # Exact substring match (highest confidence)
        exact_matches = sum(1 for kw in keywords if kw in full_query)
        if exact_matches > 0:
            return min(1.0, 0.5 + 0.2 * exact_matches)

        # Token overlap
        kw_tokens = set()
        for kw in keywords:
            kw_tokens.update(kw.replace("_", " ").split())

        overlap = len(query_tokens & kw_tokens)
        if overlap == 0:
            return 0.0

        return overlap / max(len(kw_tokens), 1)

    # ------------------------------------------------------------------
    # TF-IDF similarity
    # ------------------------------------------------------------------

    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from all keyword tokens."""
        vocab: Dict[str, int] = {}
        idx = 0
        for keywords in self._KEYWORD_MAP.values():
            for kw in keywords:
                for token in kw.replace("_", " ").split():
                    if token not in vocab:
                        vocab[token] = idx
                        idx += 1
        return vocab

    def _tfidf_vector(self, tokens: set) -> Dict[int, float]:
        """Build a sparse TF-IDF vector for a set of tokens."""
        vec: Dict[int, float] = {}
        total_docs = len(self._KEYWORD_MAP)

        for token in tokens:
            if token in self._vocab:
                idx = self._vocab[token]
                # TF: binary (present or not)
                tf = 1.0
                # IDF: log(N / df)
                df = sum(
                    1 for keywords in self._KEYWORD_MAP.values()
                    if any(token in kw for kw in keywords)
                )
                idf = math.log(total_docs / max(df, 1))
                vec[idx] = tf * idf

        return vec

    @staticmethod
    def _cosine_similarity(
        vec_a: Dict[int, float], vec_b: Dict[int, float]
    ) -> float:
        """Cosine similarity between two sparse vectors."""
        if not vec_a or not vec_b:
            return 0.0

        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not common_keys:
            return 0.0

        dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)
