"""Hard-set evaluation for normalized relevance scores, separate from soft gold.

Top-k uses reference cardinality only in the evaluator. It measures ranking, not
the model's ability to predict set size. Native set output is scored separately.
"""
import math


def _ids(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name + '_must_be_nonempty')
    if any(not isinstance(x, str) or not x for x in values):
        raise ValueError(name + '_must_contain_string_ids')
    if len(set(values)) != len(values):
        raise ValueError(name + '_has_duplicates')
    return list(values)


def evaluate_hard_set(option_ids, correct_ids, probabilities, *,
                      predicted_ids=None, tie_tolerance=1e-12):
    ids = _ids(option_ids, 'option_ids')
    gold = set(_ids(correct_ids, 'correct_ids'))
    if not gold <= set(ids):
        raise ValueError('unknown_gold_id')
    if len(ids) < 2 or not 0 <= tie_tolerance <= 1e-6:
        raise ValueError('invalid_options_or_tolerance')
    if len(probabilities) != len(ids):
        raise ValueError('probability_count_mismatch')
    p = list(probabilities)
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or not 0 <= x <= 1 for x in p):
        raise ValueError('invalid_probability')
    if not math.isclose(math.fsum(p), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError('normalized_relevance_distribution_required')
    k = len(gold)
    positive = [v for i, v in zip(ids, p) if i in gold]
    negative = [v for i, v in zip(ids, p) if i not in gold]
    mass = math.fsum(positive)
    margin = min(positive) - max(negative) if negative else None
    # A tie across the boundary cannot pass by option-order tie breaking.
    exact = float(margin > tie_tolerance) if negative else 1.0
    tv = (0.5 * math.fsum(abs(v / mass - 1 / k) for v in positive)
          if mass > 0 else None)
    balance = (1.0 if k == 1 and mass > 0 else
               1 - tv / (1 - 1 / k) if tv is not None else None)
    topn = sorted(p, reverse=True)[:k]
    topn_mass = math.fsum(topn)
    topn_tv = 0.5 * math.fsum(abs(v / topn_mass - 1 / k) for v in topn)
    topn_balance = 1.0 if k == 1 else 1 - topn_tv / (1 - 1 / k)
    q = [1 / k if i in gold else 0.0 for i in ids]
    result = {
        'target_kind': 'hard', 'label_semantics': 'set_hard',
        'probability_semantics': 'normalized_equal_membership_relevance',
        'num_options': len(ids), 'reference_cardinality': k,
        'topn_all_hit': exact, 'topn_balance': topn_balance,
        'primary_metrics': ['topn_all_hit', 'topn_balance'],
        'balance_scope': 'predicted_top_n_normalized_by_its_own_probability_mass',
        'topk_set_exact': exact, 'topk_uses_reference_cardinality': True,
        'boundary_tie': bool(negative and abs(margin) <= tie_tolerance),
        'rank_margin': margin, 'correct_set_mass': mass,
        'within_set_tv_to_uniform': tv, 'within_set_balance': balance,
        'uniform_target_brier': math.fsum((a - b) ** 2 for a, b in zip(p, q)),
        'chance_topk_exact': 1 / math.comb(len(ids), k),
    }
    if predicted_ids is not None:
        if not isinstance(predicted_ids, (list, tuple)):
            raise ValueError('predicted_ids_must_be_list')
        pred = set(predicted_ids)
        if len(pred) != len(predicted_ids) or not pred <= set(ids):
            raise ValueError('invalid_predicted_ids')
        tp = len(pred & gold)
        result.update(native_set_exact=float(pred == gold),
                      native_set_precision=tp / len(pred) if pred else 0.0,
                      native_set_recall=tp / k,
                      native_set_f1=2 * tp / (len(pred) + k),
                      cardinality_error=abs(len(pred) - k))
    return result
