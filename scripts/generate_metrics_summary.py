"""Collect accuracy, precision, recall, f1, auc for models 1,2,3,4,6 and write a single JSON summary.
This script uses only stdlib (json, csv) and does not require pandas/sklearn.
"""
import csv
import json
import os

ROOT = r"d:\ML\breast-cancer-classification\results"
out_path = os.path.join(ROOT, 'metrics_summary_models_1_2_3_4_6.json')

summary = {}

# Helper: compute binary metrics from confusion matrix [[TN,FP],[FN,TP]]
def metrics_from_cm(cm):
    TN, FP = cm[0]
    FN, TP = cm[1]
    total = TN + FP + FN + TP
    acc = (TN + TP) / total if total>0 else None
    prec = TP / (TP + FP) if (TP + FP) > 0 else None
    rec = TP / (TP + FN) if (TP + FN) > 0 else None
    f1 = (2*prec*rec/(prec+rec)) if (prec and rec and (prec+rec)>0) else None
    return acc, prec, rec, f1

# Model 1
m1 = {'sources': [], 'recomputed': False}
m1_folder = os.path.join(ROOT, 'model 1')
# read classification_report.json if present
cr_path = os.path.join(m1_folder, 'classification_report.json')
if os.path.exists(cr_path):
    m1['sources'].append(os.path.relpath(cr_path, ROOT))
    with open(cr_path, 'r', encoding='utf-8') as f:
        cr = json.load(f)
    # prefer overall accuracy and weighted avg if present
    acc = cr.get('accuracy')
    wavg = cr.get('weighted avg') or cr.get('weighted_avg')
    if wavg:
        prec = wavg.get('precision')
        rec = wavg.get('recall')
        f1 = wavg.get('f1-score') or wavg.get('f1-score')
    else:
        # fallback: compute from confusion_matrix.csv if present
        prec = rec = f1 = None
        cm_path = os.path.join(m1_folder, 'confusion_matrix.csv')
        if os.path.exists(cm_path):
            m1['sources'].append(os.path.relpath(cm_path, ROOT))
            with open(cm_path, 'r', encoding='utf-8') as cf:
                r = list(csv.reader(cf))
                try:
                    cm = [[int(x) for x in r[0]], [int(x) for x in r[1]]]
                    acc, prec, rec, f1 = metrics_from_cm(cm)
                except Exception:
                    pass
    # AUC: try to find a roc_auc in any file
    auc_val = None
    # model1 doesn't have per-sample probs; search for a roc file
    for fname in ('roc_auc.txt', 'roc_auc.json'):
        p = os.path.join(m1_folder, fname)
        if os.path.exists(p):
            m1['sources'].append(os.path.relpath(p, ROOT))
            try:
                with open(p) as f:
                    auc_val = float(f.read().strip())
            except Exception:
                pass

    m1.update({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_val})
else:
    m1.update({'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'auc': None, 'sources': []})

summary['model_1'] = m1

# Model 2 — prefer recomputed files, else predictions.csv
m2 = {'sources': [], 'recomputed': False}
m2_folder = os.path.join(ROOT, 'model 2')
re_cm = os.path.join(m2_folder, 'recomputed_confusion_matrix.csv')
re_auc = os.path.join(m2_folder, 'recomputed_roc_auc.txt')
if os.path.exists(re_cm):
    m2['sources'].append(os.path.relpath(re_cm, ROOT))
    with open(re_cm, 'r', encoding='utf-8') as f:
        r = list(csv.reader(f))
        cm = [[int(x) for x in r[0]], [int(x) for x in r[1]]]
    acc, prec, rec, f1 = metrics_from_cm(cm)
    auc_val = None
    if os.path.exists(re_auc):
        m2['sources'].append(os.path.relpath(re_auc, ROOT))
        try:
            with open(re_auc, 'r') as f:
                auc_val = float(f.read().strip())
        except Exception:
            auc_val = None
    m2.update({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_val, 'recomputed': True})
else:
    # try to read predictions.csv and compute from there (without sklearn)
    predp = os.path.join(m2_folder, 'predictions.csv')
    if os.path.exists(predp):
        m2['sources'].append(os.path.relpath(predp, ROOT))
        # parse predictions; we'll compute confusion and try to approximate AUC via simple trapezoid if needed
        rows = []
        with open(predp, 'r', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                rows.append(row)
        # build arrays
        y_true = [int(r['true_label']) for r in rows]
        y_pred = [int(r['predicted_label']) for r in rows]
        # confusion
        TN = FP = FN = TP = 0
        for t, p in zip(y_true, y_pred):
            if t==0 and p==0: TN += 1
            elif t==0 and p==1: FP += 1
            elif t==1 and p==0: FN += 1
            elif t==1 and p==1: TP += 1
        acc, prec, rec, f1 = metrics_from_cm([[TN,FP],[FN,TP]])
        # AUC: try to read malignant_prob and compute AUC via ranking method (without numpy)
        auc_val = None
        if 'malignant_prob' in rows[0]:
            # compute AUC (Mann-Whitney U / rank-sum method)
            scores = [(float(r['malignant_prob']), int(r['true_label'])) for r in rows]
            # sort by score descending
            scores.sort(key=lambda x: x[0], reverse=True)
            # compute rank sums
            n_pos = sum(1 for s in scores if s[1]==1)
            n_neg = sum(1 for s in scores if s[1]==0)
            if n_pos>0 and n_neg>0:
                # assign average ranks for ties
                ranks = []
                i = 0
                while i < len(scores):
                    j = i
                    while j < len(scores) and scores[j][0] == scores[i][0]:
                        j += 1
                    avg_rank = (i + 1 + j) / 2.0
                    for k in range(i, j):
                        ranks.append((avg_rank, scores[k][1]))
                    i = j
                # sum ranks for positive
                rank_sum_pos = sum(r for r,lab in ranks if lab==1)
                auc_val = (rank_sum_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        m2.update({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_val})
    else:
        m2.update({'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'auc': None})

summary['model_2'] = m2

# Model 3
m3 = {'sources': [], 'recomputed': False}
m3_folder = os.path.join(ROOT, 'model 3')
res3 = os.path.join(m3_folder, 'model3_ensemble_cnn_ensemble_results.json')
if os.path.exists(res3):
    m3['sources'].append(os.path.relpath(res3, ROOT))
    with open(res3, 'r', encoding='utf-8') as f:
        j = json.load(f)
    em = j.get('ensemble_metrics', {})
    acc = em.get('accuracy')
    prec = em.get('precision')
    rec = em.get('recall')
    f1 = em.get('f1')
    # try to find any numeric roc or recomputed file
    auc_val = None
    # sometimes roc_curve.png exists but no numeric AUC
    m3.update({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_val})
else:
    m3.update({'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'auc': None})

summary['model_3'] = m3

# Model 4
m4 = {'sources': [], 'recomputed': False}
m4_folder = os.path.join(ROOT, 'model 4')
res4 = os.path.join(m4_folder, 'test_metrics.json')
if os.path.exists(res4):
    m4['sources'].append(os.path.relpath(res4, ROOT))
    with open(res4, 'r', encoding='utf-8') as f:
        j = json.load(f)
    acc = j.get('accuracy')
    # prefer keys precision, recall, f1_score
    prec = j.get('precision')
    rec = j.get('recall')
    f1 = j.get('f1_score') or j.get('f1')
    auc_val = j.get('roc_auc') or j.get('roc_auc_score') or j.get('roc_auc_score')
    m4.update({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_val})
else:
    m4.update({'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'auc': None})

summary['model_4'] = m4

# Model 6
m6 = {'sources': [], 'recomputed': False}
m6_folder = os.path.join(ROOT, 'model6_handcrafted_ann_pytorch')
res6 = os.path.join(m6_folder, 'model6_metrics.csv')
if os.path.exists(res6):
    m6['sources'].append(os.path.relpath(res6, ROOT))
    with open(res6, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        row = next(rdr)
    # csv has headers accuracy,precision,recall,f1_score,roc_auc
    acc = float(row.get('accuracy')) if row.get('accuracy') else None
    prec = float(row.get('precision')) if row.get('precision') else None
    rec = float(row.get('recall')) if row.get('recall') else None
    f1 = float(row.get('f1_score') or row.get('f1')) if (row.get('f1_score') or row.get('f1')) else None
    auc_val = float(row.get('roc_auc')) if row.get('roc_auc') else None
    m6.update({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_val})
else:
    m6.update({'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'auc': None})

summary['model_6'] = m6

# Save summary JSON
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('Wrote summary to', out_path)
# Print human-readable check
for k,v in summary.items():
    print('\n', k)
    print('  accuracy:', v.get('accuracy'))
    print('  precision:', v.get('precision'))
    print('  recall:', v.get('recall'))
    print('  f1:', v.get('f1'))
    print('  auc:', v.get('auc'))
    print('  sources:', v.get('sources'))

print('\nNote: AUC is only computed/supplied when per-sample probabilities or numeric AUC were available. For models lacking per-sample predictions, AUC is null.')
