"""Deterministic, gold-blind hierarchical top-k selection.

Scores are compared only within a request. Final probabilities are conditional
on the final candidate set, never a distribution over the original catalog.
"""
import asyncio
import math
from .common import digest, dump, valid_p

PROTOCOL = 'hierarchical-choice-v1'
TOL = 1e-12


class ContextCapacityError(RuntimeError):
    pass


def ordered(ids, seed, case_key, path):
    return sorted(ids, key=lambda x: (digest([PROTOCOL, seed, case_key, path, x]), x))


def partition(ids, cap):
    """Contiguous batches, each bounded by cap; no dropped tail."""
    return [ids[i:i+cap] for i in range(0,len(ids),cap)]


def rank(ids, p, k):
    # ids were ordered with a fixed, gold-independent seed.
    ix = sorted(range(len(ids)), key=lambda i: (-p[i], i))
    tie = len(ids) > k and abs(p[ix[k-1]] - p[ix[k]]) <= TOL
    return [ids[i] for i in ix[:k]], tie


async def select_hierarchical(actor, query, *, cap=255, k=1, seed=20261015):
    """query(actor_subset, path) -> (normalized probabilities, call metadata).

This function never receives labels. k is the reference cardinality for the
existing oracle-N multiselect protocol. The API is not given gold IDs or k.
query may raise ContextCapacityError for an explicit provider size rejection.
"""
    opts = actor['options']; ids = [o['id'] for o in opts]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError('invalid_option_ids')
    if not 1 <= k < cap or k >= len(ids):
        raise ValueError('require_1_le_k_lt_cap_and_catalog_size')
    byid = dict(zip(ids, opts)); case_key = digest(actor); trace = []

    # Large tool documentation can exceed context capacity even with k+1
    # candidates. Probe the longest description plus k short descriptions
    # before expanding a costly tree. This is only a capacity check; its scores
    # never influence selection. A failed probe is reported, not truncated.
    sizes = {x:len(dump(byid[x])) for x in ids}
    longest = max(ids,key=lambda x:(sizes[x],x))
    if sizes[longest] > 32000:
        shortest = sorted((x for x in ids if x!=longest),key=lambda x:(sizes[x],x))[:k]
        members = ordered([longest,*shortest],seed,case_key,'context_probe')
        subset = {**actor,'options':[byid[x] for x in members]}
        try:
            p,meta = await query(subset,'context_probe')
            valid_p(p,len(members))
        except ContextCapacityError as exc:
            raise ContextCapacityError('minimal_cardinality_probe_exceeds_context') from exc
        trace.append(dict(path='context_probe',kind='capacity_probe',members=members,**meta))

    async def visit(members, path, depth=0):
        if depth > 32:
            raise ContextCapacityError('context_reduction_depth_limit')
        members = ordered(members, seed, case_key, path)
        if len(members) <= k:
            trace.append(dict(path=path,kind='carry',members=members,selected=members))
            return dict(selected=members, final_ids=members, final_probabilities=None,
                        boundary_tie=False)
        batch_cap = cap
        if len(members) <= cap:
            subset = {**actor, 'options':[byid[x] for x in members]}
            try:
                p, meta = await query(subset, path)
                p = valid_p(p, len(members))
                winners, tie = rank(members, p, k)
                trace.append(dict(path=path,kind='decision',members=members,
                                  probabilities=p,selected=winners,boundary_tie=tie,
                                  **meta))
                return dict(selected=winners,final_ids=members,
                            final_probabilities=p,boundary_tie=tie)
            except ContextCapacityError as exc:
                trace.append(dict(path=path,kind='context_split',members=members,
                                  reason=str(exc)))
                # With <=k+1 options any top-k reduction must compare the entire
                # remaining set. Do not silently truncate descriptions or rank
                # scores from unrelated requests to force an answer.
                if len(members) <= k+1:
                    raise ContextCapacityError('minimum_comparison_exceeds_context') from exc
                batch_cap = max(k+1, math.ceil(len(members)/2))
        batches = partition(members, batch_cap)
        children = await asyncio.gather(*(visit(g, path+'.'+str(i), depth+1)
                                          for i,g in enumerate(batches)),
                                        return_exceptions=True)
        # Drain in-flight siblings before returning an error or taking a snapshot.
        for child in children:
            if isinstance(child, BaseException):
                raise child
        survivors = [x for child in children for x in child['selected']]
        if len(survivors) >= len(members):
            raise ContextCapacityError('reduction_cannot_progress')
        return await visit(survivors, path+'.final', depth+1)

    result = await visit(ids, 'r0')
    result.update(protocol=PROTOCOL,cap=cap,k=k,seed=seed,original_option_count=len(ids),
                  probability_scope='final_survivor_set_only',
                  trace=sorted(trace,key=lambda x:x['path']),
                  intermediate_boundary_ties=sum(x.get('boundary_tie',False) for x in trace))
    return result


def score_selection(row, result):
    """Reference-dependent scoring lives entirely outside the selection path."""
    is_set = row.get('label_semantics') == 'set_hard' or row['task']=='multichoice'
    gold = set(row['correct_option_ids']) if is_set else {
        row['options'][row['gold'].index(1.0)]['id']}
    hit = set(result['selected']) == gold and not result['boundary_tie']
    final = set(result['final_ids'])
    p = result['final_probabilities']; k = result['k']
    top = sorted(p,reverse=True)[:k]; mass = math.fsum(top)
    balance = 1.0 if k==1 else 1 - (
        .5*math.fsum(abs(x/mass-1/k) for x in top))/(1-1/k)
    # This stricter sensitivity statistic flags a gold-containing cutoff tie
    # even if the fixed tie break happened to keep that gold option.
    exposed = False
    for event in result['trace']:
        if event.get('boundary_tie') and set(event['members']) & gold:
            cutoff = sorted(event['probabilities'],reverse=True)[k-1]
            exposed |= any(x in gold and abs(v-cutoff)<=TOL
                           for x,v in zip(event['members'],event['probabilities']))
    return dict(hard_accuracy=None if is_set else float(hit),
                topn_all_hit=float(hit) if is_set else None,
                final_topn_balance=balance if is_set else None,
                final_gold_survival=len(gold & final)/len(gold),
                gold_exposed_to_cutoff_tie=exposed,
                strict_tie_safe_correct=float(hit and not exposed),
                full_catalog_brier=None,full_catalog_score_mae=None)
