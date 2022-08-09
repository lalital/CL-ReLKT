import string
import re
from collections import defaultdict, Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        tx = re.sub(r'\b(a|an|the)\b.', ' ', text)
        tx = tx.replace('pad', '').replace('s', '')
        return tx

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def per_example_evaluate_with_lang(gold_answers, predictions, reference_langs=None):
    
    per_lang_scores = defaultdict(lambda: list())

    for ground_truths, prediction, lang in zip(gold_answers, predictions, reference_langs):
        total += 1

        per_lang_scores[lang]['exact_match'].append(metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths))
        per_lang_scores[lang]['f1'].append(metric_max_over_ground_truths(
          f1_score, prediction, ground_truths))

    return per_lang_scores

def per_example_evaluate(gold_answers, predictions):
 
    per_example_scores = defaultdict(lambda: list())

    for ground_truths, prediction in zip(gold_answers, predictions):

        per_example_scores['exact_match'].append(metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths))
        per_example_scores['f1'].append(metric_max_over_ground_truths(
          f1_score, prediction, ground_truths))

    return per_example_scores

def evaluate(gold_answers, predictions):
  
    f1 = exact_match = total = 0
    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

def evaluate_with_lang(gold_answers, predictions, reference_langs=None):
    per_lang_f1 = defaultdict( )
    per_lang_em = defaultdict( )
    per_lang_total = defaultdict()

    for ground_truths, prediction, lang in zip(gold_answers, predictions, reference_langs):
        per_lang_total[lang] += 1
        per_lang_em[lang] += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
        per_lang_f1[lang] += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)

    per_lang_em[lang] = 100.0 * per_lang_em[lang] / per_lang_total[lang] 
    per_lang_f1[lang] = 100.0 * per_lang_f1[lang] / per_lang_total[lang] 

    return { lang: { 'f1': per_lang_f1[lang] , 'em': per_lang_em[lang] } for lang in per_lang_total.keys() }

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}
    