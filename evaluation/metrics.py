"""
Comprehensive evaluation metrics for TextVQA task.
Includes VQA-style accuracy, BLEU, METEOR, ROUGE, F1, and LLM-as-a-Judge metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re
import string

# NLP Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# Try to download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass


# VQA-style evaluation - contractions and manual mappings from vqaEval.py
VQA_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", 
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all",
    "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've"
}

VQA_MANUAL_MAP = {
    'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'
}

VQA_ARTICLES = ['a', 'an', 'the']


class TextVQAMetrics:
    """
    Comprehensive metrics for TextVQA evaluation.
    """
    
    def __init__(self, use_llm_judge: bool = False, llm_api_key: Optional[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            use_llm_judge: Whether to use LLM-as-a-Judge evaluation
            llm_api_key: API key for LLM service (OpenAI or Anthropic)
        """
        self.use_llm_judge = use_llm_judge
        self.llm_api_key = llm_api_key
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
        
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """
        Normalize answer for fair comparison.
        
        Args:
            answer: Answer string to normalize
            
        Returns:
            Normalized answer
        """
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove punctuation
        answer = answer.translate(str.maketrans('', '', string.punctuation))
        
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer
    
    @staticmethod
    def process_punctuation_vqa(text: str) -> str:
        """
        Process punctuation following VQA evaluation style.
        
        Args:
            text: Input text
            
        Returns:
            Text with punctuation processed
        """
        # Punctuation marks to process
        punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', 
                 '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
        
        # Comma strip pattern for numbers
        comma_strip = re.compile(r"(\d)(\,)(\d)")
        # Period strip pattern
        period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        
        out_text = text
        for p in punct:
            if (p + ' ' in text or ' ' + p in text) or (comma_strip.search(text) is not None):
                out_text = out_text.replace(p, '')
            else:
                out_text = out_text.replace(p, ' ')
        
        out_text = period_strip.sub("", out_text, re.UNICODE)
        return out_text
    
    @staticmethod
    def process_digit_article_vqa(text: str) -> str:
        """
        Process digits and articles following VQA evaluation style.
        
        Args:
            text: Input text
            
        Returns:
            Text with digits and articles processed
        """
        out_text = []
        temp_text = text.lower().split()
        
        for word in temp_text:
            # Map words like "one" -> "1"
            word = VQA_MANUAL_MAP.get(word, word)
            # Remove articles
            if word not in VQA_ARTICLES:
                out_text.append(word)
        
        # Handle contractions
        for word_id, word in enumerate(out_text):
            if word in VQA_CONTRACTIONS:
                out_text[word_id] = VQA_CONTRACTIONS[word]
        
        return ' '.join(out_text)
    
    def compute_vqa_accuracy(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute VQA-style accuracy (official TextVQA metric).
        
        This follows the evaluation protocol from the TextVQA paper where
        accuracy is based on agreement with human annotators.
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers (from multiple annotators)
            
        Returns:
            Accuracy score between 0 and 1
        """
        # Clean up prediction and ground truths
        prediction = prediction.replace('\n', ' ').replace('\t', ' ').strip()
        gt_answers = [gt.replace('\n', ' ').replace('\t', ' ').strip() for gt in ground_truths]
        
        # Check if there are multiple different answers (requires processing)
        if len(set(gt_answers)) > 1:
            # Process prediction
            processed_pred = self.process_punctuation_vqa(prediction)
            processed_pred = self.process_digit_article_vqa(processed_pred)
            
            # Process ground truths
            processed_gts = []
            for gt in gt_answers:
                processed_gt = self.process_punctuation_vqa(gt)
                processed_gt = self.process_digit_article_vqa(processed_gt)
                processed_gts.append(processed_gt)
        else:
            processed_pred = prediction.lower()
            processed_gts = [gt.lower() for gt in gt_answers]
        
        # Compute accuracy based on agreement
        # For each ground truth, count how many other GTs match the prediction
        gt_acc = []
        for i, gt_answer in enumerate(processed_gts):
            # Other GT answers (excluding current one)
            other_gts = [processed_gts[j] for j in range(len(processed_gts)) if j != i]
            # Count how many match the prediction
            matching_answers = [other_gt for other_gt in other_gts if other_gt == processed_pred]
            # Accuracy is min(num_matching/3, 1) following VQA protocol
            acc = min(1.0, float(len(matching_answers)) / 3.0)
            gt_acc.append(acc)
        
        # Average accuracy across all ground truths
        if len(gt_acc) > 0:
            return float(sum(gt_acc)) / len(gt_acc)
        else:
            return 0.0
    
    def compute_exact_match(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute exact match accuracy (primary metric for TextVQA).
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            1.0 if prediction matches any ground truth, 0.0 otherwise
        """
        normalized_pred = self.normalize_answer(prediction)
        
        for gt in ground_truths:
            normalized_gt = self.normalize_answer(gt)
            if normalized_pred == normalized_gt:
                return 1.0
        
        return 0.0
    
    def compute_f1(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute token-level F1 score.
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            Maximum F1 score across all ground truths
        """
        normalized_pred = self.normalize_answer(prediction)
        pred_tokens = normalized_pred.split()
        
        if len(pred_tokens) == 0:
            return 0.0
        
        max_f1 = 0.0
        for gt in ground_truths:
            normalized_gt = self.normalize_answer(gt)
            gt_tokens = normalized_gt.split()
            
            if len(gt_tokens) == 0:
                continue
            
            # Compute precision and recall
            common_tokens = set(pred_tokens) & set(gt_tokens)
            
            if len(common_tokens) == 0:
                f1 = 0.0
            else:
                precision = len(common_tokens) / len(pred_tokens)
                recall = len(common_tokens) / len(gt_tokens)
                f1 = 2 * (precision * recall) / (precision + recall)
            
            max_f1 = max(max_f1, f1)
        
        return max_f1
    
    def compute_precision_recall(self, prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        """
        Compute precision and recall.
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            Dictionary with precision and recall scores
        """
        normalized_pred = self.normalize_answer(prediction)
        pred_tokens = normalized_pred.split()
        
        if len(pred_tokens) == 0:
            return {"precision": 0.0, "recall": 0.0}
        
        max_precision = 0.0
        max_recall = 0.0
        
        for gt in ground_truths:
            normalized_gt = self.normalize_answer(gt)
            gt_tokens = normalized_gt.split()
            
            if len(gt_tokens) == 0:
                continue
            
            common_tokens = set(pred_tokens) & set(gt_tokens)
            
            precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = len(common_tokens) / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
            
            max_precision = max(max_precision, precision)
            max_recall = max(max_recall, recall)
        
        return {"precision": max_precision, "recall": max_recall}
    
    def compute_bleu(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute BLEU score.
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            Maximum BLEU score across all ground truths
        """
        pred_tokens = self.normalize_answer(prediction).split()
        
        if len(pred_tokens) == 0:
            return 0.0
        
        max_bleu = 0.0
        for gt in ground_truths:
            gt_tokens = self.normalize_answer(gt).split()
            
            if len(gt_tokens) == 0:
                continue
            
            # Compute BLEU with smoothing
            bleu = sentence_bleu(
                [gt_tokens],
                pred_tokens,
                smoothing_function=self.smoothing.method1
            )
            max_bleu = max(max_bleu, bleu)
        
        return max_bleu
    
    def compute_meteor(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute METEOR score.
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            Maximum METEOR score across all ground truths
        """
        try:
            max_meteor = 0.0
            for gt in ground_truths:
                meteor = meteor_score([gt], prediction)
                max_meteor = max(max_meteor, meteor)
            return max_meteor
        except Exception as e:
            print(f"METEOR computation error: {e}")
            return 0.0
    
    def compute_rouge(self, prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            
        Returns:
            Dictionary with ROUGE scores
        """
        max_rouge1 = 0.0
        max_rouge2 = 0.0
        max_rougeL = 0.0
        
        for gt in ground_truths:
            scores = self.rouge_scorer.score(gt, prediction)
            max_rouge1 = max(max_rouge1, scores['rouge1'].fmeasure)
            max_rouge2 = max(max_rouge2, scores['rouge2'].fmeasure)
            max_rougeL = max(max_rougeL, scores['rougeL'].fmeasure)
        
        return {
            "rouge1": max_rouge1,
            "rouge2": max_rouge2,
            "rougeL": max_rougeL
        }
    
    def compute_llm_judge(self, prediction: str, ground_truths: List[str], question: str) -> float:
        """
        Use LLM as a judge to evaluate semantic similarity.
        
        Args:
            prediction: Predicted answer
            ground_truths: List of acceptable ground truth answers
            question: The original question
            
        Returns:
            Score from 0 to 1 indicating semantic similarity
        """
        if not self.use_llm_judge or not self.llm_api_key:
            return 0.0
        
        try:
            # Use OpenAI API
            import openai
            openai.api_key = self.llm_api_key
            
            prompt = f"""Question: {question}
Ground Truth Answers: {', '.join(ground_truths)}
Model Prediction: {prediction}

Evaluate whether the model's prediction is semantically correct compared to the ground truth answers.
Consider synonyms, paraphrasing, and equivalent meanings.
Respond with only a score from 0 to 1, where:
- 1.0 means completely correct (same meaning)
- 0.5 means partially correct
- 0.0 means completely incorrect

Score:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for question-answering systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"LLM judge error: {e}")
            return 0.0
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        ground_truths_list: List[List[str]],
        questions: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics for a set of predictions.
        
        Args:
            predictions: List of predicted answers
            ground_truths_list: List of lists of ground truth answers
            questions: Optional list of questions (for LLM judge)
            
        Returns:
            Dictionary with all metric scores
        """
        assert len(predictions) == len(ground_truths_list), \
            "Number of predictions must match number of ground truths"
        
        metrics = defaultdict(list)
        
        for i, (pred, gts) in enumerate(zip(predictions, ground_truths_list)):
            # Primary metric - VQA-style accuracy (official TextVQA metric)
            metrics['vqa_accuracy'].append(self.compute_vqa_accuracy(pred, gts))
            
            # Also compute exact match for comparison
            metrics['exact_match'].append(self.compute_exact_match(pred, gts))
            
            # Token-level metrics
            metrics['f1'].append(self.compute_f1(pred, gts))
            pr = self.compute_precision_recall(pred, gts)
            metrics['precision'].append(pr['precision'])
            metrics['recall'].append(pr['recall'])
            
            # Semantic metrics
            metrics['bleu'].append(self.compute_bleu(pred, gts))
            metrics['meteor'].append(self.compute_meteor(pred, gts))
            
            rouge = self.compute_rouge(pred, gts)
            metrics['rouge1'].append(rouge['rouge1'])
            metrics['rouge2'].append(rouge['rouge2'])
            metrics['rougeL'].append(rouge['rougeL'])
            
            # LLM judge (if enabled)
            if self.use_llm_judge and questions:
                metrics['llm_judge'].append(
                    self.compute_llm_judge(pred, gts, questions[i])
                )
        
        # Aggregate metrics
        results = {}
        for metric_name, values in metrics.items():
            results[metric_name] = np.mean(values)
            results[f"{metric_name}_std"] = np.std(values)
        
        return results
    
    def compute_per_category_metrics(
        self,
        predictions: List[str],
        ground_truths_list: List[List[str]],
        categories: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics broken down by category.
        
        Args:
            predictions: List of predicted answers
            ground_truths_list: List of lists of ground truth answers
            categories: List of category labels
            
        Returns:
            Dictionary mapping category to metrics
        """
        category_data = defaultdict(lambda: {'preds': [], 'gts': []})
        
        for pred, gts, cat in zip(predictions, ground_truths_list, categories):
            category_data[cat]['preds'].append(pred)
            category_data[cat]['gts'].append(gts)
        
        results = {}
        for cat, data in category_data.items():
            results[cat] = self.compute_all_metrics(data['preds'], data['gts'])
        
        return results


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metric scores
        title: Title for the metrics display
    """
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Primary metrics
    print(f"\n{'PRIMARY METRICS (TextVQA Official)':<40} {'Score':>10} {'Percentage':>15}")
    print("-" * 80)
    if 'vqa_accuracy' in metrics:
        print(f"{'VQA-style Accuracy (Official)':<40} {metrics['vqa_accuracy']:>10.4f} {metrics['vqa_accuracy']*100:>14.2f}%")
    if 'exact_match' in metrics:
        print(f"{'Exact Match (for comparison)':<40} {metrics['exact_match']:>10.4f} {metrics['exact_match']*100:>14.2f}%")
    
    # Semantic metrics
    print(f"\n{'SEMANTIC METRICS':<40} {'Score':>10}")
    print("-" * 80)
    for metric in ['bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL']:
        if metric in metrics:
            display_name = metric.upper().replace('ROUGE1', 'ROUGE-1').replace('ROUGE2', 'ROUGE-2').replace('ROUGEL', 'ROUGE-L')
            print(f"{display_name:<40} {metrics[metric]:>10.4f}")
    
    # Token-level metrics
    print(f"\n{'TOKEN-LEVEL METRICS':<40} {'Score':>10}")
    print("-" * 80)
    for metric in ['f1', 'precision', 'recall']:
        if metric in metrics:
            print(f"{metric.upper() if metric == 'f1' else metric.capitalize():<40} {metrics[metric]:>10.4f}")
    
    # LLM judge
    if 'llm_judge' in metrics:
        print(f"\n{'LLM-AS-A-JUDGE':<40} {'Score':>10}")
        print("-" * 80)
        print(f"{'Semantic Similarity':<40} {metrics['llm_judge']:>10.4f}")
    
    print("=" * 80)


def compute_all_metrics(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    questions: Optional[List[str]] = None,
    use_llm_judge: bool = False,
    llm_api_key: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function to compute all metrics without creating TextVQAMetrics object.
    
    Args:
        predictions: List of predicted answers
        ground_truths_list: List of lists of ground truth answers
        questions: Optional list of questions (for LLM judge)
        use_llm_judge: Whether to use LLM-as-a-Judge evaluation
        llm_api_key: API key for LLM service
        
    Returns:
        Dictionary with all metric scores
    """
    metrics_calc = TextVQAMetrics(use_llm_judge=use_llm_judge, llm_api_key=llm_api_key)
    return metrics_calc.compute_all_metrics(predictions, ground_truths_list, questions)


if __name__ == "__main__":
    # Test metrics
    print("Testing TextVQA metrics...")
    
    metrics_calc = TextVQAMetrics(use_llm_judge=False)
    
    # Test data
    predictions = [
        "nokia",
        "stop sign",
        "coca cola",
        "12:30"
    ]
    
    ground_truths_list = [
        ["nokia", "Nokia"],
        ["stop", "stop sign"],
        ["coca-cola", "coke"],
        ["12:30", "12:30 PM"]
    ]
    
    # Compute metrics
    results = metrics_calc.compute_all_metrics(predictions, ground_truths_list)
    
    # Print results
    print_metrics(results, "Test Metrics")
    
    print("\n✓ Metrics test completed successfully!")
