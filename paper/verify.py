#!/usr/bin/env python3
"""Recompute every number the MedSSL-3D paper claims, from the frozen CSVs only.

Needs five files in CSV_DIR (set it below or pass as argv[1]):
    v0_11_factorial_summary.csv
    v0_11_test_report.csv
    v0_11_test_per_patient.csv
    v0_11_val_scores.csv
    v0_11_classification_report.csv

Expected result: 66 pass / 0 fail on validation, 73 pass / 0 fail on test.
"""
import csv, sys, collections, numpy as np

CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

def load(name):
    import glob, os
    hits = glob.glob(os.path.join(CSV_DIR, f"*{name}"))
    if not hits:
        sys.exit(f"missing: {name} under {CSV_DIR}")
    return list(csv.DictReader(open(sorted(hits)[0])))

fac = load("v0_11_factorial_summary.csv")
rep = load("v0_11_test_report.csv")
cr  = load("v0_11_classification_report.csv")
pp  = load("v0_11_test_per_patient.csv")

ARMS  = ['ssl_norm', 'ssl_raw', 'scratch_norm', 'scratch_raw']
SEEDS = ['42', '123', '2024']
BOOT_SEED, BOOT_N = 20260904, 10000

OK, BAD = [], []
def chk(name, got, exp, tol=6e-4):
    ok = abs(got - exp) <= tol
    (OK if ok else BAD).append((name, got, exp))
    print(("PASS " if ok else "FAIL ") + f"{name:52s} got {got:.4f}  paper {exp:.4f}")
def sd(v): return float(np.std(v, ddof=1))
def m(v):  return float(np.mean(v))

# ───────────────────────── validation Dice ──────────────────────────────────
val = {}
for a in ARMS:
    rows = [r for r in fac if r['arm'] == a
            and r['smoke_test'].lower() in ('false', '0', '')
            and r['seed'] in SEEDS]
    d = {r['seed']: float(r['dice']) for r in rows}
    assert set(d) == set(SEEDS), (a, sorted(d))
    val[a] = d

print("== Table III: validation Dice ==")
for a, exp in [('ssl_norm', (0.5551, 0.0095)), ('ssl_raw', (0.3427, 0.0434)),
               ('scratch_norm', (0.5598, 0.0115)), ('scratch_raw', (0.3591, 0.0533))]:
    v = [val[a][s] for s in SEEDS]
    chk(f"val dice mean {a}", m(v), exp[0]); chk(f"val dice sd   {a}", sd(v), exp[1])

print("== Table IV: contrasts ==")
norm  = [(val['ssl_norm'][s] + val['scratch_norm'][s]) / 2
         - (val['ssl_raw'][s] + val['scratch_raw'][s]) / 2 for s in SEEDS]
pre   = [(val['ssl_norm'][s] + val['ssl_raw'][s]) / 2
         - (val['scratch_norm'][s] + val['scratch_raw'][s]) / 2 for s in SEEDS]
inter = [(val['ssl_norm'][s] - val['ssl_raw'][s])
         - (val['scratch_norm'][s] - val['scratch_raw'][s]) for s in SEEDS]
for i, s in enumerate(SEEDS):
    chk(f"norm effect seed {s}",     norm[i],  [0.1706, 0.2117, 0.2374][i])
    chk(f"pretrain effect seed {s}", pre[i],   [-0.0114, -0.0382, 0.0179][i])
    chk(f"interaction seed {s}",     inter[i], [0.0057, 0.0546, -0.0252][i])
chk("norm effect mean", m(norm), 0.2066);     chk("norm effect sd", sd(norm), 0.0337)
chk("pretrain effect mean", m(pre), -0.0106); chk("pretrain effect sd", sd(pre), 0.0281)

print("== VI-E variance (paper pools as the MEAN of the two within-arm SDs) ==")
pn = np.mean([sd([val['ssl_norm'][s] for s in SEEDS]),
              sd([val['scratch_norm'][s] for s in SEEDS])])
pr = np.mean([sd([val['ssl_raw'][s] for s in SEEDS]),
              sd([val['scratch_raw'][s] for s in SEEDS])])
chk("mean within-arm SD normalized", pn, 0.0105, 1e-3)
chk("mean within-arm SD raw",        pr, 0.0484, 1e-3)
chk("ratio", pr / pn, 4.6, 0.1)
chk("ratio within SSL",
    sd([val['ssl_raw'][s] for s in SEEDS]) / sd([val['ssl_norm'][s] for s in SEEDS]), 4.59, 0.02)
chk("ratio within scratch",
    sd([val['scratch_raw'][s] for s in SEEDS]) / sd([val['scratch_norm'][s] for s in SEEDS]), 4.64, 0.02)

# ───────────────────── validation classification ────────────────────────────
def crget(arm, seed, kind, key):
    r = [x for x in cr if x['arm'] == arm and x['seed'] == seed
         and x['threshold_kind'] == kind]
    assert len(r) == 1, (arm, seed, kind, len(r))
    return float(r[0][key])

print("== Table V: validation classification ==")
EXP = {'ssl_norm':     ((0.6752, 0.0182), (0.7949, 0.0438), (0.8388, 0.0302)),
       'ssl_raw':      ((0.6487, 0.0209), (0.7005, 0.0092), (0.7002, 0.0206)),
       'scratch_norm': ((0.7463, 0.0331), (0.8407, 0.0240), (0.8785, 0.0038)),
       'scratch_raw':  ((0.6151, 0.0156), (0.7005, 0.0253), (0.7198, 0.0390))}
for a in ARMS:
    b0 = [crget(a, s, 'default_0.5', 'balanced_accuracy') for s in SEEDS]
    bc = [crget(a, s, 'youden_val',  'balanced_accuracy') for s in SEEDS]
    au = [crget(a, s, 'youden_val',  'auc')               for s in SEEDS]
    chk(f"val balAcc@0.5 mean {a}", m(b0), EXP[a][0][0]); chk(f"val balAcc@0.5 sd {a}", sd(b0), EXP[a][0][1])
    chk(f"val balAcc@cal mean {a}", m(bc), EXP[a][1][0]); chk(f"val balAcc@cal sd {a}", sd(bc), EXP[a][1][1])
    chk(f"val AUC mean {a}",        m(au), EXP[a][2][0]); chk(f"val AUC sd {a}",        sd(au), EXP[a][2][1])

print("== Table VI: thresholds ==")
TH = {'ssl_norm': [0.9938, 0.9934, 0.9972], 'scratch_norm': [0.9975, 0.9715, 0.9915],
      'ssl_raw':  [0.9756, 0.5240, 0.9662], 'scratch_raw':  [0.8512, 0.7295, 0.9613]}
RANGE = {'ssl_norm': 0.0038, 'scratch_norm': 0.0259, 'ssl_raw': 0.4516, 'scratch_raw': 0.2318}
for a in ARMS:
    t = [crget(a, s, 'youden_val', 'threshold') for s in SEEDS]
    for i, s in enumerate(SEEDS):
        chk(f"threshold {a} seed {s}", t[i], TH[a][i])
    chk(f"threshold range {a}", max(t) - min(t), RANGE[a])

print(f"\nVALIDATION SUMMARY: {len(OK)} pass, {len(BAD)} fail")
for b in BAD: print("  FAILED:", b)

# ─────────────────────────────── test set ───────────────────────────────────
print("\n################ TEST SET ################")
OK.clear(); BAD.clear()
def rget(arm, seed, key):
    r = [x for x in rep if x['arm'] == arm and x['seed'] == seed]
    assert len(r) == 1, (arm, seed, len(r))
    return float(r[0][key])

print("== Table VII: test Dice + regions ==")
E = {'ssl_norm':     (0.5439, 0.0084, 0.8084, 0.0108, 0.6998, 0.0190, 0.5413, 0.0114),
     'ssl_raw':      (0.3221, 0.0304, 0.6789, 0.0092, 0.4794, 0.0256, 0.3307, 0.0038),
     'scratch_norm': (0.5526, 0.0069, 0.8125, 0.0011, 0.7092, 0.0075, 0.5554, 0.0076),
     'scratch_raw':  (0.3366, 0.0463, 0.6807, 0.0421, 0.4935, 0.0344, 0.3242, 0.0311)}
tdice = {}
for a in ARMS:
    for i, (k, lab) in enumerate([('dice', 'Dice'), ('wt', 'WT'), ('tc', 'TC'), ('et', 'ET')]):
        v = [rget(a, s, k) for s in SEEDS]
        if k == 'dice': tdice[a] = v
        chk(f"test {lab} mean {a}", m(v), E[a][2*i]); chk(f"test {lab} sd   {a}", sd(v), E[a][2*i+1])

print("== bootstrap CIs ==")
def per_patient(arm, key='dice'):
    d = collections.defaultdict(list)
    for r in pp:
        if r['arm'] == arm and r[key] not in ('', 'nan'):
            d[r['patient_id']].append(float(r[key]))
    return [float(np.mean(v)) for v in d.values()]
CI = {'ssl_norm': (0.4967, 0.5893), 'ssl_raw': (0.2739, 0.3684),
      'scratch_norm': (0.5046, 0.5979), 'scratch_raw': (0.2884, 0.3835)}
for a in ARMS:
    arr = np.asarray(per_patient(a), float)
    rs = np.random.RandomState(BOOT_SEED)
    mm = arr[rs.randint(0, arr.size, (BOOT_N, arr.size))].mean(1)
    chk(f"boot lo {a}", float(np.percentile(mm, 2.5)),  CI[a][0], 1e-3)
    chk(f"boot hi {a}", float(np.percentile(mm, 97.5)), CI[a][1], 1e-3)

print("== test classification ==")
EC = {'ssl_norm': (0.7704, 0.0378, 0.8511, 0.0379), 'ssl_raw': (0.7537, 0.0370, 0.8370, 0.0206),
      'scratch_norm': (0.7704, 0.0210, 0.8793, 0.0324), 'scratch_raw': (0.6944, 0.0729, 0.7556, 0.0395)}
for a in ARMS:
    b = [rget(a, s, 'balanced_accuracy') for s in SEEDS]
    u = [rget(a, s, 'auc') for s in SEEDS]
    chk(f"test balAcc mean {a}", m(b), EC[a][0]); chk(f"test balAcc sd {a}", sd(b), EC[a][1])
    chk(f"test AUC mean {a}",    m(u), EC[a][2]); chk(f"test AUC sd {a}",    sd(u), EC[a][3])
print("   verdicts:", dict(collections.Counter(r['verdict'] for r in rep)))

print("== derived claims ==")
chk("test norm effect (SSL)",     m(tdice['ssl_norm'])     - m(tdice['ssl_raw']),     0.2218)
chk("test norm effect (scratch)", m(tdice['scratch_norm']) - m(tdice['scratch_raw']), 0.2160)
chk("scratch lead, normalized",   m(tdice['scratch_norm']) - m(tdice['ssl_norm']),    0.0087)
chk("scratch lead, raw",          m(tdice['scratch_raw'])  - m(tdice['ssl_raw']),     0.0145)
for a, e in [('ssl_norm', -0.0112), ('ssl_raw', -0.0206),
             ('scratch_norm', -0.0072), ('scratch_raw', -0.0225)]:
    chk(f"val->test drop {a}", m(tdice[a]) - m([val[a][s] for s in SEEDS]), e)
def regmean(k, cond):
    return m([m([rget(a, s, k) for s in SEEDS]) for a in cond])
for k, e in [('wt', 0.19), ('tc', 0.45), ('et', 0.67)]:
    nn = regmean(k, ['ssl_norm', 'scratch_norm']); rr = regmean(k, ['ssl_raw', 'scratch_raw'])
    chk(f"relative gain {k.upper()}", (nn - rr) / rr, e, 0.005)
for cond, e_m, e_s in [(('ssl_norm', 'ssl_raw'), 0.1386, 0.0394),
                       (('scratch_norm', 'scratch_raw'), 0.1587, 0.0368)]:
    d = [crget(cond[0], s, 'youden_val', 'auc') - crget(cond[1], s, 'youden_val', 'auc') for s in SEEDS]
    chk(f"val AUC norm effect {cond[0]}", m(d), e_m); chk(f"val AUC norm effect sd {cond[0]}", sd(d), e_s)
d = [crget('ssl_norm', s, 'youden_val', 'auc') - crget('scratch_norm', s, 'youden_val', 'auc') for s in SEEDS]
chk("val AUC ssl-scratch (norm) mean", m(d), -0.0397)
chk("val AUC ssl-scratch (norm) sd",   sd(d), 0.0294)

print("== threshold transfer (6 better / 4 worse / 2 unchanged, delta +0.019) ==")
def balacc(y, p, t):
    y = np.array(y); pr = (np.array(p) >= t).astype(int)
    return (((pr == 1) & (y == 1)).sum() / (y == 1).sum()
            + ((pr == 0) & (y == 0)).sum() / (y == 0).sum()) / 2
froz, naive, better, worse, same = [], [], 0, 0, 0
for a in ARMS:
    for s in SEEDS:
        rows = [r for r in pp if r['arm'] == a and r['seed'] == s]
        y = [int(r['y_true']) for r in rows]; p = [float(r['p_hgg']) for r in rows]
        t = float(rows[0]['threshold'])
        bf, bn = balacc(y, p, t), balacc(y, p, 0.5)
        froz.append(bf); naive.append(bn)
        better += bf > bn + 1e-12; worse += bf < bn - 1e-12; same += abs(bf - bn) <= 1e-12
chk("frozen-threshold mean balAcc", m(froz), 0.7472)
chk("naive-0.5 mean balAcc",        m(naive), 0.7282)
chk("transfer delta",               m(froz) - m(naive), 0.0190)
print(f"   per-run: better {better}, worse {worse}, unchanged {same}  (paper: 6 / 4 / 2)")

print(f"\nTEST SUMMARY: {len(OK)} pass, {len(BAD)} fail")
for b in BAD: print("  FAILED:", b)
