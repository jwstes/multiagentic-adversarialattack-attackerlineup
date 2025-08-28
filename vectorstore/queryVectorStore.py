import os
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from collections import defaultdict
import textwrap
import re
import json
import time

# ----------------- Config -----------------
QDRANT_URL = "https://5cf05368-4850-4f7a-8912-f2af8b6b788c.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.rQVnGexdv6IYTOvcOEN5Q7EEo9Vq3wuQCwdibak7eIw"
COLLECTION_NAME = "ART-Toolbox-AttackStrategies"

EMBED_MODEL_NAME = "all-mpnet-base-v2"   # must match your index embeddings
RERANKER_MODEL = "BAAI/bge-reranker-large"  # heavy local reranker; fits your GPU

INIT_K = 80      # initial top-k per query (larger for recall)
FUSED_K = 150    # after fusion
RERANK_KEEP = 80 # keep this many into reranker
FINAL_K = 12     # final contexts for LLM
MMR_LAMBDA = 0.7

# Detector/forensics synonyms (expand aggressively)
DETECTOR_TERMS = [
    "deepfake detection", "ai-generated image detection", "synthetic image detection",
    "image forensics", "forensic detector", "forgery detection", "manipulation detection",
    "authenticity detection", "gan image detector", "diffusion detector",
    "face manipulation detection", "face forgery detection", "fake image detector",
    "deepfake detector", "real vs fake detection", "splicing detection",
    "PRNU", "noise residuals", "CNN-based detector", "transformer-based detector",
    "GAN fingerprint", "spectral artifacts", "forensic traces"
]

# Method family terms (helpful for query variants/reranking context)
METHOD_FAMILIES = [
    "white-box", "black-box", "transfer-based", "query-efficient", "decision-based",
    "score-based", "gradient-based", "ensemble attack",
    "L_inf", "L∞", "L2", "L0", "LPIPS", "perceptual constraint", "SSIM",
    "imperceptible", "small perturbation", "physical attack", "digital attack",
]

# ------------------------------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL)

def summarize_image_facts(facts: List[Dict[str, Any]]) -> str:
    cues = []
    for f in facts:
        area = f.get("area", "")
        concl = f.get("conclusion", "")
        reason = f.get("reasoning", "")
        if concl:
            cues.append(f"{area}: {concl}")
        elif reason:
            cues.append(f"{area}: {reason}")
    cue_text = "; ".join(cues[:25])
    return textwrap.shorten(
        f"Image forensic cues suggest possible AI-generation. Goal: evade AI-generated image/deepfake detector with minimal, imperceptible perturbations. Cues: {cue_text}",
        width=1000, placeholder="..."
    )

def build_core_queries(image_summary: str, available_methods: List[str]) -> List[str]:
    method_str = ", ".join(available_methods)
    base = [
        f"Adversarial attacks to evade AI-generated image/deepfake detectors (image forensics). Prefer imperceptible small Lp noise, both white-box and black-box. Methods available: {method_str}.",
        f"Best attacks for forensic/manipulation detectors (CNN/Transformer). Consider L_inf/L2 constraints, transferability, and query-efficient decision/score-based methods. Methods: {method_str}.",
        f"Attacks applicable to detection tasks (not just classifiers): decision-based black-box, score-based black-box, and gradient-based white-box options. Methods: {method_str}.",
        f"Given forensic cues: {image_summary}. Choose attacks that reliably evade deepfake/synthetic image detectors under small Lp budgets. Which of {method_str} fit best?"
    ]
    # Expand with detector synonyms
    expanded = [
        f"Evading {{term}} with adversarial examples; imperceptible, small L_inf/L2; consider white/black-box and transfer-based. Methods: {method_str}."
        .replace("{term}", t) for t in DETECTOR_TERMS
    ]
    # Add method family variants (helps dense retrieval recall)
    family_q = [
        f"Adversarial methods for {fam} to fool AI-generated image/forensic detectors. Methods: {method_str}."
        for fam in METHOD_FAMILIES
    ]
    return base + expanded + family_q

def method_probe_queries(available_methods: List[str]) -> List[str]:
    probes = []
    for m in available_methods:
        for term in DETECTOR_TERMS:
            probes.append(f"{m} attack for {term}: applicability, norms (L_inf/L2/L0/LPIPS), threat model (white/black-box), transferability, query complexity.")
    return probes

def encode(texts: List[str]) -> np.ndarray:
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def qdrant_search(query: str, k: int = INIT_K, query_filter: models.Filter | None = None) -> List[Dict[str, Any]]:
    qv = encode([query])[0]
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=qv,
        limit=k,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter
    )
    return [{
        "id": h.id,
        "score": float(h.score),
        "payload": h.payload,
        "text": h.payload.get("text", "")
    } for h in hits]

def rrf_fuse(result_lists: List[List[Dict[str, Any]]], k: int = FUSED_K, k_rrf: int = 60) -> List[Dict[str, Any]]:
    scores = defaultdict(float)
    id2doc = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            scores[doc["id"]] += 1.0 / (k_rrf + rank)
            id2doc[doc["id"]] = doc
    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
    return [id2doc[i] for i in ranked_ids]

def cross_encoder_rerank(query: str, docs: List[Dict[str, Any]], keep: int = RERANK_KEEP) -> List[Dict[str, Any]]:
    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, s in ranked[:min(keep, len(ranked))]]

def keyword_boost(docs: List[Dict[str, Any]], positive_terms: List[str], boost: float = 0.25) -> List[Dict[str, Any]]:
    # Adds a small bonus for detector/forensics lexical matches (post-rerank tie-breaker)
    terms = [re.escape(t.lower()) for t in positive_terms]
    if not terms:
        return docs
    pattern = re.compile("|".join(terms))
    rescored = []
    for rank, d in enumerate(docs):
        text = d["text"].lower()
        num_hits = len(pattern.findall(text))
        # Attach a score proxy for tie-breaking that respects initial order
        d2 = dict(d)
        d2["_kw_boost"] = num_hits * boost
        d2["_rank"] = rank
        rescored.append(d2)
    rescored.sort(key=lambda x: (x["_kw_boost"], -x["_rank"]), reverse=True)
    return rescored

def mmr_select(query: str, docs: List[Dict[str, Any]], final_k: int = FINAL_K, lambda_mult: float = MMR_LAMBDA) -> List[Dict[str, Any]]:
    q_vec = encode([query])[0]
    d_vecs = encode([d["text"] for d in docs])
    selected, selected_idx, candidates = [], [], list(range(len(docs)))
    sim_to_query = d_vecs @ q_vec
    sim_between = d_vecs @ d_vecs.T
    while len(selected) < min(final_k, len(docs)) and candidates:
        if not selected:
            idx = int(np.argmax(sim_to_query[candidates]))
            idx = candidates[idx]
        else:
            best_idx, best_score = None, -1e9
            for c in candidates:
                diversity = max(sim_between[c, selected_idx]) if selected_idx else 0.0
                mmr = lambda_mult * sim_to_query[c] - (1 - lambda_mult) * diversity
                if mmr > best_score:
                    best_score, best_idx = mmr, c
            idx = best_idx
        selected.append(docs[idx])
        selected_idx.append(idx)
        candidates.remove(idx)
    return selected

# Optional: Use this filter only if you run the post-tagging step below to set detector_related=True
def make_detector_filter() -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="detector_related",
                match=models.MatchValue(value=True)
            )
        ]
    )

def retrieve_contexts(available_methods: List[str], image_facts: List[Dict[str, Any]], use_filter: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    image_summary = summarize_image_facts(image_facts)
    core_queries = build_core_queries(image_summary, available_methods)
    probes = method_probe_queries(available_methods)
    all_queries = core_queries + probes

    qfilter = make_detector_filter() if use_filter else None

    # Dense search for each query
    result_sets = [qdrant_search(q, k=INIT_K, query_filter=qfilter) for q in all_queries]

    # RRF fuse
    fused = rrf_fuse(result_sets, k=FUSED_K)

    # Rerank with strong cross-encoder
    anchor_query = core_queries[0]
    reranked = cross_encoder_rerank(anchor_query, fused, keep=RERANK_KEEP)

    # Keyword boost toward detector/forensics mentions
    boosted = keyword_boost(reranked, DETECTOR_TERMS, boost=0.3)

    # MMR for diversity
    final_contexts = mmr_select(anchor_query, boosted, final_k=FINAL_K, lambda_mult=MMR_LAMBDA)
    return anchor_query, final_contexts

def startQuery(imageAnalysis):
    from misc import attackMethods
    available_methods = attackMethods.all_methods
    image_facts = json.loads(open("output/analysis.json", "r").read())

    print("[Vector Store] Querying Vector Store...")
    sT = time.time()
    query_str, contexts = retrieve_contexts(available_methods, image_facts, use_filter=False)
    eT = time.time()

    formattedContext = ""
    for i, c in enumerate(contexts, 1):
        formattedContext += f"\n[{i}]\n"
        formattedContext += textwrap.shorten(c["text"], width=1000, placeholder="...")
    
    print("[Vector Store] Vector Store Queried. It took (seconds)", eT - sT)

    file = open('output/vectorStoreResults.txt', 'w')
    file.writelines(formattedContext)
    file.close()
    
    return formattedContext

















