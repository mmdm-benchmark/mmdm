"""Metrics for one native-plus-grouped evaluation, with explicit denominators."""
import collections
from .common import valid_p, gold_kind
from .set_metrics import evaluate_hard_set
from .grouped_choice import score_selection

def label_kind(r):
    return 'hard_set' if r.get('label_semantics')=='set_hard' or r['task']=='multichoice' else gold_kind(r['gold'])

def score(r,rec):
    kind=label_kind(r);v=dict(id=r['id'],category=r['task_category'],kind=kind,task=r['task'],valid=rec.get('ok',False),
        mode=rec['mode'],status=rec['status'],correct=None,brier=None,tv=None,score_mae=None,score_normalized_mae=None,
        native_set_balance=None,grouped_final_balance=None)
    if not v['valid']:return v
    if rec['mode']=='grouped':
        m=score_selection(r,rec['result']);v['correct']=m['topn_all_hit'] if kind=='hard_set' else m['hard_accuracy']
        v['grouped_final_balance']=m['final_topn_balance'];return v
    p=valid_p(rec['probabilities'],len(r['gold']));q=r['gold']
    if kind=='hard_set':
        m=evaluate_hard_set([o['id'] for o in r['options']],r['correct_option_ids'],p)
        v['correct']=m['topn_all_hit'];v['native_set_balance']=m['topn_balance']
    elif kind=='hard':
        winners=[i for i,x in enumerate(p) if abs(x-max(p))<=1e-12]
        v['correct']=float(len(winners)==1 and q[winners[0]]==1)
    v['brier']=sum((a-b)**2 for a,b in zip(p,q));v['tv']=.5*sum(abs(a-b) for a,b in zip(p,q))
    if r['task']=='score':
        vs=r['score_values'];v['score_mae']=abs(sum((a-b)*s for a,b,s in zip(p,q,vs)))
        v['score_normalized_mae']=v['score_mae']/(max(vs)-min(vs))
    return v

def summarize(vs):
    good=[v for v in vs if v['valid']]
    def group(kinds):
        allv=[v for v in vs if v['kind'] in kinds];valid=[v for v in allv if v['valid']]
        correct=sum(v['correct'] for v in valid) if kinds!={'soft'} else None
        return dict(rows=len(allv),valid=len(valid),correct=correct,
                    accuracy_valid=correct/len(valid) if valid and correct is not None else None,
                    accuracy_all=correct/len(allv) if allv and correct is not None else None)
    def metric(name,subset):
        vals=[v[name] for v in subset if v[name] is not None]
        return dict(n=len(vals),mean=sum(vals)/len(vals) if vals else None)
    soft=[v for v in good if v['kind']=='soft'];scores=[v for v in good if v['task']=='score']
    return dict(rows=len(vs),valid=len(good),coverage=len(good)/len(vs) if vs else None,
                hard_single=group({'hard'}),hard_set=group({'hard_set'}),soft=group({'soft'}),hard_combined=group({'hard','hard_set'}),
                soft_brier=metric('brier',soft),soft_tv=metric('tv',soft),score_mae=metric('score_mae',scores),
                score_normalized_mae=metric('score_normalized_mae',scores),native_set_balance=metric('native_set_balance',good),
                grouped_final_balance=metric('grouped_final_balance',good),
                modes=dict(collections.Counter(v['mode'] for v in vs)),
                invalid_statuses=dict(collections.Counter(v['status'] for v in vs if not v['valid'])))
