"""Numbers added to the ICSPIS version that are not in verify.py.
Recomputed from the frozen per-patient test CSV (no model, no GPU, test set not reopened).
    python verify_icspis_extra.py <results dir>
"""
import csv, sys, collections
from scipy.stats import wilcoxon, binomtest
D = sys.argv[1] if len(sys.argv) > 1 else 'results'
rows = list(csv.DictReader(open(D + '/v0_11_test_per_patient.csv')))
acc = collections.defaultdict(list); grade = {}
for r in rows:
    acc[(r['arm'], r['patient_id'])].append(float(r['dice'])); grade[r['patient_id']] = int(r['y_true'])
mean = {k: sum(v) / len(v) for k, v in acc.items()}
pids = sorted(grade)
out = {}
for cond in ('ssl', 'scratch'):
    d = [mean[(cond + '_norm', p)] - mean[(cond + '_raw', p)] for p in pids]
    w = wilcoxon(d)
    worse = [(p, round(x, 4), 'HGG' if grade[p] else 'LGG', round(mean[(cond + '_raw', p)], 4), round(mean[(cond + '_norm', p)], 4))
             for p, x in zip(pids, d) if x < 0]
    out[cond] = dict(n=len(d), improved=sum(x > 0 for x in d), wilcoxon_W=w.statistic, wilcoxon_p=w.pvalue,
                     sign_p=binomtest(sum(x > 0 for x in d), len(d)).pvalue, worse=worse)
for k, v in out.items():
    print(k, 'n', v['n'], 'improved', v['improved'], 'W', v['wilcoxon_W'], 'p_wilcoxon %.3g' % v['wilcoxon_p'], 'p_sign %.3g' % v['sign_p'])
    for x in v['worse']: print('   worse:', x)
print('LGG in test:', sum(1 for p in pids if grade[p] == 0), 'of', len(pids))
