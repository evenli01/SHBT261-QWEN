import re
import evaluate
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import Counter, defaultdict
import torch  # needed for semantic similarity


# =========================
# Metric loaders (global)
# =========================
try:
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
except Exception as e:
    print(f"Warning: Failed to load some metrics: {e}")
    bleu_metric = None
    meteor_metric = None
    rouge_metric = None
    semantic_model = None


# =========================
# Text normalization helpers
# =========================

def preprocess_answer(answer: str) -> str:
    """
    Standard VQA-style text preprocessing.
    """
    answer = str(answer).lower()
    answer = answer.replace("\n", " ")
    answer = answer.replace("\t", " ")
    answer = answer.strip()
    return answer


def tokenize_answer(answer: str):
    """
    Simple whitespace tokenization after VQA-style preprocessing.
    """
    return preprocess_answer(answer).split()


# =========================
# Official VQA-style accuracy
# =========================

def compute_vqa_accuracy(ground_truth_list, predicted_answer):
    """
    Compute VQA-style consensus accuracy.

    Official metric:
        acc = min(1, (number of matching human answers) / 3)

    Matching is **exact string match** after normalization.
    """
    if not ground_truth_list:
        return 0.0

    predicted_answer = preprocess_answer(predicted_answer)
    ground_truth_list = [preprocess_answer(ans) for ans in ground_truth_list]

    match_count = sum(1 for gt in ground_truth_list if gt == predicted_answer)
    return min(1.0, match_count / 3.0)


# =========================
# Token-level Precision / Recall / F1
# =========================

def token_overlap_prf1(predicted_answer: str, ground_truth: str):
    """
    Compute token-level precision, recall, and F1 between two strings.

    We treat answers as bags of tokens, and compute:
        precision = |intersection| / |pred_tokens|
        recall    = |intersection| / |gt_tokens|
        F1        = 2 * P * R / (P + R)

    If there is no overlap or one side is empty, all scores are 0.
    """
    pred_tokens = tokenize_answer(predicted_answer)
    gt_tokens = tokenize_answer(ground_truth)

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        # both empty -> treat as perfect match
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    common = pred_counter & gt_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / float(len(pred_tokens))
    recall = num_same / float(len(gt_tokens))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def best_token_prf1_over_ground_truths(ground_truth_list, predicted_answer):
    """
    For multiple ground-truth answers, compute token-level precision/recall/F1
    against each GT and return the best (highest F1) triple.
    """
    if not ground_truth_list:
        return 0.0, 0.0, 0.0

    best_p, best_r, best_f1 = 0.0, 0.0, 0.0
    for gt in ground_truth_list:
        p, r, f1 = token_overlap_prf1(predicted_answer, gt)
        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = r
    return best_p, best_r, best_f1


# =========================
# Semantic Similarity
# =========================

def compute_semantic_similarity(ground_truth_list, predicted_answer):
    """
    Compute semantic similarity between predicted answer and ground truth answers.
    Returns the maximum cosine similarity score among all ground truth answers.
    """
    if semantic_model is None:
        return 0.0

    # Encode
    pred_emb = semantic_model.encode(predicted_answer, convert_to_tensor=True)
    gt_embs = semantic_model.encode(ground_truth_list, convert_to_tensor=True)

    cosine_scores = util.cos_sim(pred_emb, gt_embs)
    return float(torch.max(cosine_scores).item()) if cosine_scores.numel() > 0 else 0.0


# =========================
# Question-type classification (for per-category metrics)
# =========================

def classify_question(question: str) -> str:
    """
    Heuristic question-type classifier, matching the logic you use in run_eval.py.

    Returns one of:
        "brand", "number", "date", "time", "text", "general"
    """
    q = (question or "").lower()

    brand_keywords = ["brand", "logo", "label", "company", "manufacturer", "maker"]
    if any(k in q for k in brand_keywords):
        return "brand"

    number_keywords = ["number", "percent", "%", "price", "$", "cost", "amount", "how much", "how many"]
    if any(k in q for k in number_keywords):
        return "number"

    date_keywords = ["year", "date", "born", "since", "when"]
    if any(k in q for k in date_keywords):
        return "date"

    time_keywords = ["time", "clock", "what time"]
    if any(k in q for k in time_keywords):
        return "time"

    text_keywords = ["text", "say", "spell", "read", "what does", "what is written"]
    if any(k in q for k in text_keywords):
        return "text"

    return "general"


# =========================
# Aggregate metric computation
# =========================

def calculate_metrics(results):
    """
    Calculate average metrics over the results.

    Each item in `results` is expected to have at least:
        - 'predicted_answer'  (str)
        - 'ground_truth_answers' (list[str])

    Optional for per-category metrics:
        - 'q_type' (string question type; if missing we infer from 'question')
        - 'question' (used to infer q_type when needed)
    """
    if not results:
        return {}

    total_acc = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    predictions = []
    references = []
    semantic_scores = []

    # Per-category containers: {q_type: {"acc": [...], "f1": [...], "precision": [...], "recall": [...]} }
    category_stats = defaultdict(lambda: {"acc": [], "f1": [], "precision": [], "recall": []})

    for item in results:
        pred = item.get("predicted_answer", "")
        gts = item.get("ground_truth_answers", [])

        # -------------------------
        # 1) Official VQA accuracy
        # -------------------------
        acc = compute_vqa_accuracy(gts, pred)
        total_acc += acc

        # -------------------------
        # 2) Token-level P / R / F1
        # -------------------------
        p, r, f1 = best_token_prf1_over_ground_truths(gts, pred)
        total_precision += p
        total_recall += r
        total_f1 += f1

        # -------------------------
        # 3) Per-category stats
        # -------------------------
        q_type = item.get("q_type")
        if not q_type:
            q_type = classify_question(item.get("question", ""))
        category_stats[q_type]["acc"].append(acc)
        category_stats[q_type]["precision"].append(p)
        category_stats[q_type]["recall"].append(r)
        category_stats[q_type]["f1"].append(f1)

        # -------------------------
        # 4) Text for BLEU/METEOR/ROUGE
        # -------------------------
        predictions.append(pred)
        references.append(gts)

        # -------------------------
        # 5) Semantic similarity
        # -------------------------
        if semantic_model:
            try:
                sim = compute_semantic_similarity(gts, pred)
                semantic_scores.append(sim)
            except Exception as e:
                print(f"Error computing semantic similarity: {e}")

    n = float(len(results))

    metrics = {
        # Official primary metric
        "accuracy": total_acc / n,

        # New token-level metrics
        "precision_token": total_precision / n,
        "recall_token": total_recall / n,
        "f1_token": total_f1 / n,
    }

    # -------------------------
    # BLEU
    # -------------------------
    if bleu_metric is not None:
        try:
            bleu_score = bleu_metric.compute(
                predictions=predictions,
                references=references,
                max_order=2,  # bigrams; answers are short
                smooth=True,
            )
            metrics["bleu"] = bleu_score["bleu"]
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            metrics["bleu"] = 0.0

    # -------------------------
    # METEOR
    # -------------------------
    if meteor_metric is not None:
        try:
            meteor_score = meteor_metric.compute(
                predictions=predictions,
                references=references,
            )
            metrics["meteor"] = meteor_score["meteor"]
        except Exception as e:
            print(f"Error computing METEOR: {e}")
            metrics["meteor"] = 0.0

    # -------------------------
    # ROUGE
    # -------------------------
    if rouge_metric is not None:
        try:
            rouge_score = rouge_metric.compute(
                predictions=predictions,
                references=references,
            )
            metrics["rouge1"] = rouge_score["rouge1"]
            metrics["rouge2"] = rouge_score["rouge2"]
            metrics["rougeL"] = rouge_score["rougeL"]
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            metrics["rouge1"] = 0.0
            metrics["rouge2"] = 0.0
            metrics["rougeL"] = 0.0

    # -------------------------
    # Semantic similarity
    # -------------------------
    if semantic_scores:
        metrics["semantic_similarity"] = float(sum(semantic_scores) / len(semantic_scores))
    else:
        metrics["semantic_similarity"] = 0.0

    # -------------------------
    # Per-category metrics
    # -------------------------
    per_category = {}
    for q_type, stats in category_stats.items():
        def safe_mean(x_list):
            return float(sum(x_list) / len(x_list)) if x_list else 0.0

        per_category[q_type] = {
            "accuracy": safe_mean(stats["acc"]),
            "precision_token": safe_mean(stats["precision"]),
            "recall_token": safe_mean(stats["recall"]),
            "f1_token": safe_mean(stats["f1"]),
        }

    metrics["per_category"] = per_category

    return metrics
