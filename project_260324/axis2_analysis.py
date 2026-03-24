"""
Axis 2: Phase별 프롬프트 특성 차이 (7개 분석)
Output: axis2_results.png, axis2_stats.csv
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from scipy.spatial.distance import cosine
from scipy.stats import linregress

np.random.seed(42)

# Font
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

FI_COLOR = '#2196F3'; FD_COLOR = '#F44336'
IDE_COLOR = '#4CAF50'; FIN_COLOR = '#FF9800'
OUT_DIR = '/home/user/prompt-analysis_03/project_260324/'

with open(OUT_DIR + 'preprocess_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

df = cache['df']
part_df = cache['part_df']
embeddings = cache['embeddings']
umap_coords = cache['umap_coords']
CONTEXT_KEYWORDS = cache['CONTEXT_KEYWORDS']
OBJECT_KEYWORDS = cache['OBJECT_KEYWORDS']

stats_rows = []

def mwu(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2: return np.nan, np.nan, np.nan
    U, p = mannwhitneyu(a, b, alternative='two-sided')
    r = 1 - 2*U/(len(a)*len(b))
    return U, p, r

def bootstrap_ci(a, b, n=1000):
    np.random.seed(42)
    a, b = a.dropna().values, b.dropna().values
    d = [np.random.choice(a, len(a), True).mean() - np.random.choice(b, len(b), True).mean() for _ in range(n)]
    return np.percentile(d, 2.5), np.percentile(d, 97.5)

def add_stats(ax, U, p, r, n1, n2):
    sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
    ax.text(0.02, 0.98, f'U={U:.0f} p={p:.4f}{sig}\nr={r:.3f} n={n1},{n2}',
            transform=ax.transAxes, va='top', fontsize=6.5, family='monospace',
            bbox=dict(boxstyle='round', fc='wheat', alpha=.5))

def record(name, a, b, test_type='MWU'):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if test_type == 'MWU':
        U, p, r = mwu(a, b)
        ci = bootstrap_ci(a, b)
        stats_rows.append({'분석': name, 'test': 'Mann-Whitney U', 'U/W': U, 'p': p,
                           'effect_size': r, 'CI_lo': ci[0], 'CI_hi': ci[1],
                           'n1': len(a), 'n2': len(b), 'mean1': a.mean(), 'mean2': b.mean()})
    elif test_type == 'Wilcoxon':
        if len(a) < 2: return
        try:
            W, p = wilcoxon(a, b)
            n = len(a)
            r = W / (n*(n+1)/2)
        except:
            W, p, r = np.nan, np.nan, np.nan
        stats_rows.append({'분석': name, 'test': 'Wilcoxon', 'U/W': W, 'p': p,
                           'effect_size': r, 'CI_lo': np.nan, 'CI_hi': np.nan,
                           'n1': len(a), 'n2': len(b), 'mean1': a.mean(), 'mean2': b.mean()})

# ── Phase-level data ──
ide_df = df[df['phase'] == 'ideation']
fin_df = df[df['phase'] == 'final']

# Phase × Group
ide_fi = ide_df[ide_df['participant_group'] == 'FI']
ide_fd = ide_df[ide_df['participant_group'] == 'FD']
fin_fi = fin_df[fin_df['participant_group'] == 'FI']
fin_fd = fin_df[fin_df['participant_group'] == 'FD']

# ── Participant-phase level aggregation ──
phase_part = []
for pid, grp in df.groupby('participant_id'):
    r0 = grp.iloc[0]
    for phase in ['ideation', 'final']:
        pg = grp[grp['phase'] == phase]
        if len(pg) == 0: continue
        idx = pg.index.tolist()
        emb = embeddings[idx]
        fl_dist = cosine(emb[0], emb[-1]) if len(emb) > 1 else 0
        if len(emb) > 2:
            dists = [cosine(emb[i-1], emb[i]) for i in range(1, len(emb))]
            slope = linregress(range(len(dists)), dists).slope
        else:
            slope = 0
        phase_part.append({
            'participant_id': pid, 'participant_group': r0['participant_group'],
            'geft_score': r0['geft_score'], 'phase': phase,
            'turn_count': len(pg), 'char_count': pg['char_count'].mean(),
            'ctx_ratio': pg['ctx_ratio'].mean(),
            'ctx_count': pg['ctx_count'].mean(), 'obj_count': pg['obj_count'].mean(),
            'first_last_distance': fl_dist, 'divergence_slope': slope,
        })
pp = pd.DataFrame(phase_part)

# ═══════════════════════════════
fig = plt.figure(figsize=(28, 40))
fig.suptitle('Axis 2: Phase별 프롬프트 특성 차이', fontsize=20, fontweight='bold', y=0.995)

# ── 2-1: Phase UMAP ──
ax = fig.add_subplot(10, 3, 1)
for phase, mk in [('ideation', 'o'), ('final', '^')]:
    for grp, clr in [('FI', FI_COLOR), ('FD', FD_COLOR)]:
        mask = (df['phase'] == phase) & (df['participant_group'] == grp)
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1], c=clr, marker=mk,
                   s=25, alpha=.6, label=f'{grp}-{phase}')
ax.legend(fontsize=6); ax.set_title('2-1(a) UMAP: Phase×FI/FD')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

ax = fig.add_subplot(10, 3, 2)
mask_ide = df['phase'] == 'ideation'
for grp, clr, mk in [('FI', FI_COLOR, 'o'), ('FD', FD_COLOR, '^')]:
    m = mask_ide & (df['participant_group'] == grp)
    ax.scatter(umap_coords[m, 0], umap_coords[m, 1], c=clr, marker=mk, s=25, alpha=.6, label=grp)
ax.legend(fontsize=8); ax.set_title('2-1(b) UMAP: Ideation')

ax = fig.add_subplot(10, 3, 3)
mask_fin = df['phase'] == 'final'
for grp, clr, mk in [('FI', FI_COLOR, 'o'), ('FD', FD_COLOR, '^')]:
    m = mask_fin & (df['participant_group'] == grp)
    ax.scatter(umap_coords[m, 0], umap_coords[m, 1], c=clr, marker=mk, s=25, alpha=.6, label=grp)
ax.legend(fontsize=8); ax.set_title('2-1(c) UMAP: Final')

# Centroid distances
for phase in ['ideation', 'final']:
    m_fi = (df['phase'] == phase) & (df['participant_group'] == 'FI')
    m_fd = (df['phase'] == phase) & (df['participant_group'] == 'FD')
    c_fi = umap_coords[m_fi].mean(axis=0)
    c_fd = umap_coords[m_fd].mean(axis=0)
    dist = np.linalg.norm(c_fi - c_fd)
    stats_rows.append({'분석': f'2-1 UMAP centroid({phase})', 'test': 'Euclidean', 'U/W': dist,
                       'p': np.nan, 'effect_size': np.nan, 'CI_lo': np.nan, 'CI_hi': np.nan,
                       'n1': m_fi.sum(), 'n2': m_fd.sum(), 'mean1': np.nan, 'mean2': np.nan})

# ── 2-2: Phase keyword analysis ──
ax = fig.add_subplot(10, 3, 4)
groups = ['FI-ideation', 'FD-ideation', 'FI-final', 'FD-final']
data_list = [pp[(pp['participant_group']=='FI')&(pp['phase']=='ideation')]['ctx_ratio'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='ideation')]['ctx_ratio'],
             pp[(pp['participant_group']=='FI')&(pp['phase']=='final')]['ctx_ratio'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='final')]['ctx_ratio']]
colors = [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR]
bp = ax.boxplot([d.dropna() for d in data_list], labels=['FI-ide', 'FD-ide', 'FI-fin', 'FD-fin'],
                patch_artist=True)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c + '40')
for i, (d, c) in enumerate(zip(data_list, colors)):
    d = d.dropna()
    ax.scatter(np.random.normal(i+1, .04, len(d)), d, c=c, s=15, alpha=.6)
ax.set_ylabel('맥락 비율'); ax.set_title('2-2(a) Phase×FI/FD 맥락비율')

# MWU for each phase
for phase in ['ideation', 'final']:
    fi_v = pp[(pp['participant_group']=='FI')&(pp['phase']==phase)]['ctx_ratio']
    fd_v = pp[(pp['participant_group']=='FD')&(pp['phase']==phase)]['ctx_ratio']
    record(f'2-2 맥락비율 {phase} FI vs FD', fi_v, fd_v)

# Wilcoxon: same participant ideation vs final
paired_data = []
for pid in df['participant_id'].unique():
    ide_v = pp[(pp['participant_id']==pid)&(pp['phase']=='ideation')]['ctx_ratio']
    fin_v = pp[(pp['participant_id']==pid)&(pp['phase']=='final')]['ctx_ratio']
    if len(ide_v) > 0 and len(fin_v) > 0:
        paired_data.append((ide_v.values[0], fin_v.values[0]))
if paired_data:
    ide_arr, fin_arr = zip(*paired_data)
    record('2-2 맥락비율 ideation vs final (paired)', pd.Series(ide_arr), pd.Series(fin_arr), 'Wilcoxon')

# (b, c) ctx/obj count bars
for kw_i, (kw, kw_name) in enumerate([('ctx_count', '맥락'), ('obj_count', '객체')]):
    ax = fig.add_subplot(10, 3, 5 + kw_i)
    x = np.arange(2)
    for gi, (grp, clr, off) in enumerate([('FI', FI_COLOR, -.15), ('FD', FD_COLOR, .15)]):
        vals = [pp[(pp['participant_group']==grp)&(pp['phase']==ph)][kw].mean() for ph in ['ideation', 'final']]
        errs = [pp[(pp['participant_group']==grp)&(pp['phase']==ph)][kw].sem()*1.96 for ph in ['ideation', 'final']]
        ax.bar(x+off, vals, .3, yerr=errs, label=grp, color=clr, alpha=.8, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(['ideation', 'final'])
    ax.legend(fontsize=8); ax.set_ylabel(f'평균 {kw_name} 키워드 수')
    ax.set_title(f'2-2({"b" if kw_i==0 else "c"}) Phase×FI/FD {kw_name} 키워드 수')

# (d) Phase GEFT × ctx_ratio scatter
ax = fig.add_subplot(10, 3, 7)
for phase, clr, mk in [('ideation', IDE_COLOR, 'o'), ('final', FIN_COLOR, '^')]:
    sub = pp[pp['phase'] == phase]
    ax.scatter(sub['geft_score'], sub['ctx_ratio'], c=clr, marker=mk, s=30, alpha=.6, label=phase)
    mask = sub['ctx_ratio'].notna()
    if mask.sum() > 2:
        z = np.polyfit(sub.loc[mask, 'geft_score'], sub.loc[mask, 'ctx_ratio'], 1)
        xl = np.linspace(sub['geft_score'].min(), sub['geft_score'].max(), 50)
        ax.plot(xl, np.polyval(z, xl), c=clr, ls='--', alpha=.5)
ax.legend(fontsize=8); ax.set_xlabel('GEFT 점수'); ax.set_ylabel('맥락 비율')
ax.set_title('2-2(d) Phase별 GEFT × 맥락비율')

# ── 2-3: Phase char count ──
ax = fig.add_subplot(10, 3, 8)
data_list = [pp[(pp['participant_group']=='FI')&(pp['phase']=='ideation')]['char_count'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='ideation')]['char_count'],
             pp[(pp['participant_group']=='FI')&(pp['phase']=='final')]['char_count'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='final')]['char_count']]
bp = ax.boxplot([d.dropna() for d in data_list], labels=['FI-ide', 'FD-ide', 'FI-fin', 'FD-fin'],
                patch_artist=True)
for patch, c in zip(bp['boxes'], [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR]):
    patch.set_facecolor(c + '40')
for i, (d, c) in enumerate(zip(data_list, [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR])):
    d = d.dropna()
    ax.scatter(np.random.normal(i+1, .04, len(d)), d, c=c, s=15, alpha=.6)
ax.set_ylabel('평균 글자 수'); ax.set_title('2-3 Phase별 글자 수')

for phase in ['ideation', 'final']:
    fi_v = pp[(pp['participant_group']=='FI')&(pp['phase']==phase)]['char_count']
    fd_v = pp[(pp['participant_group']=='FD')&(pp['phase']==phase)]['char_count']
    record(f'2-3 글자수 {phase} FI vs FD', fi_v, fd_v)
# Paired Wilcoxon
paired = []
for pid in df['participant_id'].unique():
    i_v = pp[(pp['participant_id']==pid)&(pp['phase']=='ideation')]['char_count']
    f_v = pp[(pp['participant_id']==pid)&(pp['phase']=='final')]['char_count']
    if len(i_v)>0 and len(f_v)>0: paired.append((i_v.values[0], f_v.values[0]))
if paired:
    record('2-3 글자수 ideation vs final (paired)', pd.Series([x[0] for x in paired]), pd.Series([x[1] for x in paired]), 'Wilcoxon')

# ── 2-4: Phase turn count ──
ax = fig.add_subplot(10, 3, 9)
data_list = [pp[(pp['participant_group']=='FI')&(pp['phase']=='ideation')]['turn_count'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='ideation')]['turn_count'],
             pp[(pp['participant_group']=='FI')&(pp['phase']=='final')]['turn_count'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='final')]['turn_count']]
bp = ax.boxplot([d.dropna() for d in data_list], labels=['FI-ide', 'FD-ide', 'FI-fin', 'FD-fin'],
                patch_artist=True)
for patch, c in zip(bp['boxes'], [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR]):
    patch.set_facecolor(c + '40')
for i, (d, c) in enumerate(zip(data_list, [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR])):
    d = d.dropna()
    ax.scatter(np.random.normal(i+1, .04, len(d)), d, c=c, s=15, alpha=.6)
ax.set_ylabel('턴 수'); ax.set_title('2-4 Phase별 턴 수')

for phase in ['ideation', 'final']:
    fi_v = pp[(pp['participant_group']=='FI')&(pp['phase']==phase)]['turn_count']
    fd_v = pp[(pp['participant_group']=='FD')&(pp['phase']==phase)]['turn_count']
    record(f'2-4 턴수 {phase} FI vs FD', fi_v, fd_v)

# Phase transition point
transition_turns = []
for pid, grp in df.groupby('participant_id'):
    ide = grp[grp['phase']=='ideation']
    if len(ide) > 0:
        transition_turns.append({
            'participant_id': pid,
            'participant_group': grp.iloc[0]['participant_group'],
            'transition_turn': len(ide)
        })
tt_df = pd.DataFrame(transition_turns)
ax2 = fig.add_subplot(10, 3, 10)
fi_tt = tt_df[tt_df['participant_group']=='FI']['transition_turn']
fd_tt = tt_df[tt_df['participant_group']=='FD']['transition_turn']
bp = ax2.boxplot([fi_tt, fd_tt], labels=['FI', 'FD'], patch_artist=True)
bp['boxes'][0].set_facecolor(FI_COLOR+'40'); bp['boxes'][1].set_facecolor(FD_COLOR+'40')
ax2.scatter(np.random.normal(1,.04,len(fi_tt)), fi_tt, c=FI_COLOR, s=20, alpha=.6)
ax2.scatter(np.random.normal(2,.04,len(fd_tt)), fd_tt, c=FD_COLOR, s=20, alpha=.6)
ax2.set_ylabel('전환 시점 (턴)'); ax2.set_title('2-4 Phase 전환 시점')
record('2-4 전환시점 FI vs FD', fi_tt, fd_tt)

# ── 2-5: Phase first-last distance ──
ax = fig.add_subplot(10, 3, 11)
data_list = [pp[(pp['participant_group']=='FI')&(pp['phase']=='ideation')]['first_last_distance'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='ideation')]['first_last_distance'],
             pp[(pp['participant_group']=='FI')&(pp['phase']=='final')]['first_last_distance'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='final')]['first_last_distance']]
bp = ax.boxplot([d.dropna() for d in data_list], labels=['FI-ide', 'FD-ide', 'FI-fin', 'FD-fin'],
                patch_artist=True)
for patch, c in zip(bp['boxes'], [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR]):
    patch.set_facecolor(c + '40')
for i, (d, c) in enumerate(zip(data_list, [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR])):
    d = d.dropna()
    ax.scatter(np.random.normal(i+1, .04, len(d)), d, c=c, s=15, alpha=.6)
ax.set_ylabel('첫-마지막 거리'); ax.set_title('2-5 Phase별 첫-마지막 턴 거리')

for phase in ['ideation', 'final']:
    fi_v = pp[(pp['participant_group']=='FI')&(pp['phase']==phase)]['first_last_distance']
    fd_v = pp[(pp['participant_group']=='FD')&(pp['phase']==phase)]['first_last_distance']
    record(f'2-5 첫-마지막 {phase} FI vs FD', fi_v, fd_v)

# ── 2-6: Phase transition keyword change ──
def count_kw(text, keywords):
    return sum(str(text).count(kw) for kw in keywords)

transition_data = []
for pid, grp in df.groupby('participant_id'):
    ide_grp = grp[grp['phase']=='ideation']
    fin_grp = grp[grp['phase']=='final']
    if len(ide_grp)==0 or len(fin_grp)==0: continue
    # Last ideation, first final
    ide_last = ide_grp.iloc[-1]
    fin_first = fin_grp.iloc[0]
    ide_cr = ide_last['ctx_ratio'] if pd.notna(ide_last['ctx_ratio']) else 0
    fin_cr = fin_first['ctx_ratio'] if pd.notna(fin_first['ctx_ratio']) else 0
    # Keyword frequencies for top keywords
    ide_kw = {}
    fin_kw = {}
    for kw in CONTEXT_KEYWORDS + OBJECT_KEYWORDS:
        ide_c = str(ide_last['prompt_raw']).count(kw)
        fin_c = str(fin_first['prompt_raw']).count(kw)
        if ide_c > 0: ide_kw[kw] = ide_c
        if fin_c > 0: fin_kw[kw] = fin_c
    transition_data.append({
        'participant_id': pid,
        'participant_group': grp.iloc[0]['participant_group'],
        'ctx_change': fin_cr - ide_cr,
        'ide_kw': ide_kw, 'fin_kw': fin_kw,
    })
tr_df = pd.DataFrame(transition_data)

# (a) ctx_change boxplot
ax = fig.add_subplot(10, 3, 12)
fi_cc = tr_df[tr_df['participant_group']=='FI']['ctx_change']
fd_cc = tr_df[tr_df['participant_group']=='FD']['ctx_change']
bp = ax.boxplot([fi_cc.dropna(), fd_cc.dropna()], labels=['FI', 'FD'], patch_artist=True)
bp['boxes'][0].set_facecolor(FI_COLOR+'40'); bp['boxes'][1].set_facecolor(FD_COLOR+'40')
ax.scatter(np.random.normal(1,.04,len(fi_cc)), fi_cc, c=FI_COLOR, s=20, alpha=.6)
ax.scatter(np.random.normal(2,.04,len(fd_cc)), fd_cc, c=FD_COLOR, s=20, alpha=.6)
ax.axhline(0, color='gray', ls='--', alpha=.5)
ax.set_ylabel('맥락비율 변화량'); ax.set_title('2-6(a) Phase 전환 시 맥락비율 변화')
record('2-6 맥락비율변화 FI vs FD', fi_cc, fd_cc)

# (b, c) Top 10 keyword heatmaps
for gi, (grp, grp_name) in enumerate([('FI', 'FI'), ('FD', 'FD')]):
    ax = fig.add_subplot(10, 3, 13 + gi)
    sub = tr_df[tr_df['participant_group'] == grp]
    # Aggregate keyword freq
    all_ide = {}; all_fin = {}
    for _, row in sub.iterrows():
        for k, v in row['ide_kw'].items(): all_ide[k] = all_ide.get(k, 0) + v
        for k, v in row['fin_kw'].items(): all_fin[k] = all_fin.get(k, 0) + v
    # Union top 10
    all_kws = set(list(sorted(all_ide, key=all_ide.get, reverse=True))[:10] +
                  list(sorted(all_fin, key=all_fin.get, reverse=True))[:10])
    if not all_kws:
        ax.text(.5, .5, '데이터 없음', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'2-6({"b" if gi==0 else "c"}) {grp_name} 키워드 변화')
        continue
    kw_list = sorted(all_kws)
    hm = pd.DataFrame({
        'ideation': [all_ide.get(k, 0) for k in kw_list],
        'final': [all_fin.get(k, 0) for k in kw_list],
    }, index=kw_list)
    sns.heatmap(hm, annot=True, fmt='d', cmap='YlGnBu', ax=ax, cbar=False)
    ax.set_title(f'2-6({"b" if gi==0 else "c"}) {grp_name} Top 키워드 변화')

# ── 2-7: Phase divergence/convergence change ──
# (a) Phase divergence boxplot
ax = fig.add_subplot(10, 3, 15)
data_list = [pp[(pp['participant_group']=='FI')&(pp['phase']=='ideation')]['first_last_distance'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='ideation')]['first_last_distance'],
             pp[(pp['participant_group']=='FI')&(pp['phase']=='final')]['first_last_distance'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='final')]['first_last_distance']]
bp = ax.boxplot([d.dropna() for d in data_list], labels=['FI-ide', 'FD-ide', 'FI-fin', 'FD-fin'],
                patch_artist=True)
for patch, c in zip(bp['boxes'], [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR]):
    patch.set_facecolor(c + '40')
for i, (d, c) in enumerate(zip(data_list, [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR])):
    d = d.dropna()
    ax.scatter(np.random.normal(i+1, .04, len(d)), d, c=c, s=15, alpha=.6)
ax.set_ylabel('발산 정도'); ax.set_title('2-7(a) Phase별 발산 정도')

# (b) Phase convergence slope boxplot
ax = fig.add_subplot(10, 3, 16)
data_list = [pp[(pp['participant_group']=='FI')&(pp['phase']=='ideation')]['divergence_slope'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='ideation')]['divergence_slope'],
             pp[(pp['participant_group']=='FI')&(pp['phase']=='final')]['divergence_slope'],
             pp[(pp['participant_group']=='FD')&(pp['phase']=='final')]['divergence_slope']]
bp = ax.boxplot([d.dropna() for d in data_list], labels=['FI-ide', 'FD-ide', 'FI-fin', 'FD-fin'],
                patch_artist=True)
for patch, c in zip(bp['boxes'], [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR]):
    patch.set_facecolor(c + '40')
for i, (d, c) in enumerate(zip(data_list, [FI_COLOR, FD_COLOR, FI_COLOR, FD_COLOR])):
    d = d.dropna()
    ax.scatter(np.random.normal(i+1, .04, len(d)), d, c=c, s=15, alpha=.6)
ax.set_ylabel('수렴 기울기'); ax.set_title('2-7(b) Phase별 수렴 기울기')

# (c) ideation→final divergence change FI vs FD
ax = fig.add_subplot(10, 3, 17)
div_change = []
for pid in df['participant_id'].unique():
    ide_v = pp[(pp['participant_id']==pid)&(pp['phase']=='ideation')]['first_last_distance']
    fin_v = pp[(pp['participant_id']==pid)&(pp['phase']=='final')]['first_last_distance']
    if len(ide_v)>0 and len(fin_v)>0:
        pgrp = df[df['participant_id']==pid].iloc[0]['participant_group']
        div_change.append({'pid': pid, 'group': pgrp, 'change': fin_v.values[0] - ide_v.values[0]})
dc_df = pd.DataFrame(div_change)
if len(dc_df) > 0:
    fi_dc = dc_df[dc_df['group']=='FI']['change']
    fd_dc = dc_df[dc_df['group']=='FD']['change']
    bp = ax.boxplot([fi_dc.dropna(), fd_dc.dropna()], labels=['FI', 'FD'], patch_artist=True)
    bp['boxes'][0].set_facecolor(FI_COLOR+'40'); bp['boxes'][1].set_facecolor(FD_COLOR+'40')
    ax.scatter(np.random.normal(1,.04,len(fi_dc)), fi_dc, c=FI_COLOR, s=20, alpha=.6)
    ax.scatter(np.random.normal(2,.04,len(fd_dc)), fd_dc, c=FD_COLOR, s=20, alpha=.6)
    ax.axhline(0, color='gray', ls='--', alpha=.5)
    record('2-7 발산변화량 FI vs FD', fi_dc, fd_dc)
ax.set_ylabel('발산 변화량'); ax.set_title('2-7(c) 발산 변화량 FI vs FD')

# Wilcoxon for phase changes
paired_div = []
paired_conv = []
for pid in df['participant_id'].unique():
    ide_d = pp[(pp['participant_id']==pid)&(pp['phase']=='ideation')]['first_last_distance']
    fin_d = pp[(pp['participant_id']==pid)&(pp['phase']=='final')]['first_last_distance']
    ide_s = pp[(pp['participant_id']==pid)&(pp['phase']=='ideation')]['divergence_slope']
    fin_s = pp[(pp['participant_id']==pid)&(pp['phase']=='final')]['divergence_slope']
    if len(ide_d)>0 and len(fin_d)>0: paired_div.append((ide_d.values[0], fin_d.values[0]))
    if len(ide_s)>0 and len(fin_s)>0: paired_conv.append((ide_s.values[0], fin_s.values[0]))

if paired_div:
    record('2-7 발산 ideation vs final (paired)',
           pd.Series([x[0] for x in paired_div]), pd.Series([x[1] for x in paired_div]), 'Wilcoxon')
if paired_conv:
    record('2-7 수렴 ideation vs final (paired)',
           pd.Series([x[0] for x in paired_conv]), pd.Series([x[1] for x in paired_conv]), 'Wilcoxon')

plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(OUT_DIR + 'axis2_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("axis2_results.png saved.")

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(OUT_DIR + 'axis2_stats.csv', index=False, encoding='utf-8-sig')
print("axis2_stats.csv saved.")
print(stats_df.to_string())
