"""
Axis 1: GEFT 점수에 따른 프롬프트 특성 (10개 분석)
Output: axis1_results.png, axis1_stats.csv
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact
from scipy.spatial.distance import cosine
from scipy.stats import linregress

np.random.seed(42)

# ── Font setup ──
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── Colors ──
FI_COLOR = '#2196F3'
FD_COLOR = '#F44336'

OUT_DIR = '/home/user/prompt-analysis_03/project_260324/'

# ── Load preprocessed data ──
with open(OUT_DIR + 'preprocess_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

df = cache['df']
part_df = cache['part_df']
embeddings = cache['embeddings']
umap_coords = cache['umap_coords']

fi_part = part_df[part_df['participant_group'] == 'FI']
fd_part = part_df[part_df['participant_group'] == 'FD']

# ── Helper functions ──
def mann_whitney_test(fi_vals, fd_vals):
    fi_vals = fi_vals.dropna()
    fd_vals = fd_vals.dropna()
    if len(fi_vals) < 2 or len(fd_vals) < 2:
        return np.nan, np.nan, np.nan
    U, p = mannwhitneyu(fi_vals, fd_vals, alternative='two-sided')
    n1, n2 = len(fi_vals), len(fd_vals)
    r_rb = 1 - (2 * U) / (n1 * n2)  # rank-biserial r
    return U, p, r_rb

def spearman_test(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    return spearmanr(x[mask], y[mask])

def bootstrap_ci(fi_vals, fd_vals, n_boot=1000, ci=0.95):
    np.random.seed(42)
    fi_vals = fi_vals.dropna().values
    fd_vals = fd_vals.dropna().values
    diffs = []
    for _ in range(n_boot):
        fi_s = np.random.choice(fi_vals, len(fi_vals), replace=True)
        fd_s = np.random.choice(fd_vals, len(fd_vals), replace=True)
        diffs.append(fi_s.mean() - fd_s.mean())
    alpha = (1 - ci) / 2
    return np.percentile(diffs, alpha * 100), np.percentile(diffs, (1 - alpha) * 100)

def add_stats_text(ax, U, p, r_rb, n_fi, n_fd):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(0.02, 0.98, f'U={U:.0f}, p={p:.4f} {sig}\nr_rb={r_rb:.3f}, n(FI)={n_fi}, n(FD)={n_fd}',
            transform=ax.transAxes, va='top', fontsize=7, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

def boxplot_with_points(ax, fi_vals, fd_vals, ylabel, title):
    data = [fi_vals.dropna(), fd_vals.dropna()]
    bp = ax.boxplot(data, labels=['FI', 'FD'], patch_artist=True,
                    boxprops=dict(facecolor='white'),
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(FI_COLOR + '40')
    bp['boxes'][1].set_facecolor(FD_COLOR + '40')
    # Jitter points
    for i, (vals, color) in enumerate(zip(data, [FI_COLOR, FD_COLOR])):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(x, vals, alpha=0.6, color=color, s=20, zorder=5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    U, p, r_rb = mann_whitney_test(fi_vals, fd_vals)
    add_stats_text(ax, U, p, r_rb, len(fi_vals.dropna()), len(fd_vals.dropna()))
    return U, p, r_rb

def scatter_geft(ax, geft, values, ylabel, title):
    mask = ~(np.isnan(geft) | np.isnan(values))
    ax.scatter(geft[mask], values[mask], alpha=0.6, c='#333', s=30)
    r_s, p_s = spearman_test(geft.values, values.values)
    # Regression line
    if mask.sum() > 2:
        z = np.polyfit(geft[mask], values[mask], 1)
        x_line = np.linspace(geft[mask].min(), geft[mask].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5)
    ax.set_xlabel('GEFT 점수')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sig = '***' if p_s < 0.001 else '**' if p_s < 0.01 else '*' if p_s < 0.05 else 'ns'
    ax.text(0.02, 0.98, f'Spearman r={r_s:.3f}, p={p_s:.4f} {sig}',
            transform=ax.transAxes, va='top', fontsize=7, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))
    return r_s, p_s

# ── Stats collector ──
stats_rows = []

def record_stat(name, fi_vals, fd_vals, geft, values):
    U, p, r_rb = mann_whitney_test(fi_vals, fd_vals)
    ci_lo, ci_hi = bootstrap_ci(fi_vals, fd_vals)
    r_s, p_s = spearman_test(geft.values, values.values)
    stats_rows.append({
        '변인': name,
        'FI mean': fi_vals.mean(),
        'FI SD': fi_vals.std(),
        'FD mean': fd_vals.mean(),
        'FD SD': fd_vals.std(),
        'Mann-Whitney U': U,
        'p': p,
        'rank-biserial r': r_rb,
        'Bootstrap 95% CI lower': ci_lo,
        'Bootstrap 95% CI upper': ci_hi,
        'Spearman r (GEFT)': r_s,
        'Spearman p': p_s,
    })

# ═══════════════════════════════════════════════════
# Create figure: very large multi-panel figure
# Layout: 10 analyses, each with 2-4 subplots
# We'll use a grid layout
# ═══════════════════════════════════════════════════

fig = plt.figure(figsize=(28, 48))
fig.suptitle('Axis 1: GEFT 점수에 따른 프롬프트 특성', fontsize=20, fontweight='bold', y=0.995)

# ── Analysis 1-1: UMAP Clustering ──
# Find optimal k
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(umap_coords)
    sil_scores.append(silhouette_score(umap_coords, labels))

best_k = K_range[np.argmax(sil_scores)]
best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = best_km.fit_predict(umap_coords)

# 1-1 (a) Silhouette
ax = fig.add_subplot(12, 4, 1)
ax.plot(list(K_range), sil_scores, 'bo-')
ax.axvline(best_k, color='red', ls='--', alpha=0.7)
ax.set_xlabel('k'); ax.set_ylabel('실루엣 스코어')
ax.set_title(f'1-1(a) 실루엣 스코어 (최적 k={best_k})')
ax.text(0.5, 0.02, f'Best silhouette={max(sil_scores):.3f}',
        transform=ax.transAxes, ha='center', fontsize=8)

# 1-1 (b) UMAP FI/FD
ax = fig.add_subplot(12, 4, 2)
fi_mask = df['participant_group'] == 'FI'
fd_mask = df['participant_group'] == 'FD'
ax.scatter(umap_coords[fi_mask, 0], umap_coords[fi_mask, 1], c=FI_COLOR, marker='o', s=25, alpha=0.7, label='FI')
ax.scatter(umap_coords[fd_mask, 0], umap_coords[fd_mask, 1], c=FD_COLOR, marker='^', s=25, alpha=0.7, label='FD')
ax.legend(fontsize=8); ax.set_title('1-1(b) UMAP: FI vs FD')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

# 1-1 (c) UMAP clusters
ax = fig.add_subplot(12, 4, 3)
for k_i in range(best_k):
    mask = cluster_labels == k_i
    ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1], s=25, alpha=0.7, label=f'C{k_i+1}')
ax.legend(fontsize=7, ncol=2); ax.set_title(f'1-1(c) UMAP: {best_k} 클러스터')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

# 1-1 (d) UMAP GEFT continuous
ax = fig.add_subplot(12, 4, 4)
sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=df['geft_score'], cmap='viridis', s=25, alpha=0.7)
plt.colorbar(sc, ax=ax, label='GEFT 점수')
ax.set_title('1-1(d) UMAP: GEFT 점수')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

# ── Analysis 1-2: Context/Object Keywords ──
# (a) Grouped bar chart
ax = fig.add_subplot(12, 4, 5)
fi_prompt = df[df['participant_group'] == 'FI']
fd_prompt = df[df['participant_group'] == 'FD']

means = pd.DataFrame({
    'FI': [fi_prompt['ctx_count'].mean(), fi_prompt['obj_count'].mean()],
    'FD': [fd_prompt['ctx_count'].mean(), fd_prompt['obj_count'].mean()]
}, index=['맥락 키워드', '객체 키워드'])
sems = pd.DataFrame({
    'FI': [fi_prompt['ctx_count'].sem() * 1.96, fi_prompt['obj_count'].sem() * 1.96],
    'FD': [fd_prompt['ctx_count'].sem() * 1.96, fd_prompt['obj_count'].sem() * 1.96]
}, index=['맥락 키워드', '객체 키워드'])
x = np.arange(2)
ax.bar(x - 0.15, means['FI'], 0.3, yerr=sems['FI'], label='FI', color=FI_COLOR, alpha=0.8, capsize=3)
ax.bar(x + 0.15, means['FD'], 0.3, yerr=sems['FD'], label='FD', color=FD_COLOR, alpha=0.8, capsize=3)
ax.set_xticks(x); ax.set_xticklabels(means.index)
ax.legend(fontsize=8); ax.set_ylabel('평균 키워드 수')
ax.set_title('1-2(a) FI/FD별 맥락·객체 키워드 수')

# (b) ctx_ratio boxplot (participant-level)
ax = fig.add_subplot(12, 4, 6)
boxplot_with_points(ax, fi_part['mean_ctx_ratio'], fd_part['mean_ctx_ratio'],
                    '맥락 비율', '1-2(b) FI/FD별 맥락 비율')
record_stat('맥락비율(mean_ctx_ratio)', fi_part['mean_ctx_ratio'], fd_part['mean_ctx_ratio'],
            part_df['geft_score'], part_df['mean_ctx_ratio'])

# (c) Turn-by-turn keyword trends
ax = fig.add_subplot(12, 4, 7)
for grp_name, color, ls in [('FI', FI_COLOR, '-'), ('FD', FD_COLOR, '--')]:
    sub = df[df['participant_group'] == grp_name].copy()
    # Normalize turn number within participant
    sub['turn_num'] = sub.groupby('participant_id').cumcount() + 1
    for kw_type, marker in [('ctx_count', 'o'), ('obj_count', 's')]:
        trend = sub.groupby('turn_num')[kw_type].mean()
        trend = trend[trend.index <= 10]
        label = f'{grp_name} {"맥락" if kw_type == "ctx_count" else "객체"}'
        ax.plot(trend.index, trend.values, color=color, ls=ls if kw_type == 'ctx_count' else ':',
                marker=marker, markersize=4, alpha=0.8, label=label)
ax.set_xlabel('턴 번호'); ax.set_ylabel('평균 키워드 수')
ax.set_title('1-2(c) 턴별 맥락/객체 키워드 추이')
ax.legend(fontsize=6, ncol=2)

# (d) GEFT × ctx_ratio scatter
ax = fig.add_subplot(12, 4, 8)
scatter_geft(ax, part_df['geft_score'], part_df['mean_ctx_ratio'],
             '맥락 비율', '1-2(d) GEFT × 맥락 비율')

# ── Analysis 1-3 ~ 1-9: Standard boxplot + scatter pairs ──
analyses = [
    ('1-3', 'total_char_count', '전체 프롬프트 글자 수'),
    ('1-4', 'initial_length', '초기 프롬프트 길이'),
    ('1-5', 'mean_refinement_length', '평균 수정 프롬프트 길이'),
    ('1-6', 'turn_count', '턴 수'),
    ('1-7', 'first_last_distance', '첫-마지막 턴 거리'),
    ('1-8', 'total_change', '의미 총 변화량'),
    ('1-9', 'ref_img_count', '참조 이미지 사용 횟수'),
]

for i, (num, col, label) in enumerate(analyses):
    # Boxplot
    ax1 = fig.add_subplot(12, 4, 9 + i * 2)
    boxplot_with_points(ax1, fi_part[col], fd_part[col], label, f'{num}(a) FI/FD별 {label}')
    record_stat(label, fi_part[col], fd_part[col], part_df['geft_score'], part_df[col])

    # GEFT scatter
    ax2 = fig.add_subplot(12, 4, 10 + i * 2)
    scatter_geft(ax2, part_df['geft_score'], part_df[col], label, f'{num}(b) GEFT × {label}')

# ── Analysis 1-10: Divergence/Convergence ──
# (a) Divergence boxplot
ax = fig.add_subplot(12, 4, 23)
boxplot_with_points(ax, fi_part['first_last_distance'], fd_part['first_last_distance'],
                    '발산 정도', '1-10(a) FI/FD별 발산 정도')

# (b) Convergence slope boxplot
ax = fig.add_subplot(12, 4, 24)
boxplot_with_points(ax, fi_part['divergence_slope'], fd_part['divergence_slope'],
                    '수렴 기울기', '1-10(b) FI/FD별 수렴 기울기')
record_stat('발산(first_last_distance)', fi_part['first_last_distance'], fd_part['first_last_distance'],
            part_df['geft_score'], part_df['first_last_distance'])
record_stat('수렴(divergence_slope)', fi_part['divergence_slope'], fd_part['divergence_slope'],
            part_df['geft_score'], part_df['divergence_slope'])

# (c) Divergence × Convergence scatter with quadrants
ax = fig.add_subplot(12, 4, 25)
div_med = part_df['first_last_distance'].median()
conv_med = part_df['divergence_slope'].median()

for _, row in part_df.iterrows():
    color = FI_COLOR if row['participant_group'] == 'FI' else FD_COLOR
    marker = 'o' if row['participant_group'] == 'FI' else '^'
    ax.scatter(row['first_last_distance'], row['divergence_slope'], c=color, marker=marker, s=40, alpha=0.7)

ax.axhline(conv_med, color='gray', ls='--', alpha=0.5)
ax.axvline(div_med, color='gray', ls='--', alpha=0.5)
# Quadrant labels
ax.text(0.95, 0.95, '지속적 탐색형', transform=ax.transAxes, ha='right', va='top', fontsize=7, color='gray')
ax.text(0.05, 0.95, '점진적 발산형', transform=ax.transAxes, ha='left', va='top', fontsize=7, color='gray')
ax.text(0.95, 0.05, '탐색 후 안착형', transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color='gray')
ax.text(0.05, 0.05, '안정적 정교화형', transform=ax.transAxes, ha='left', va='bottom', fontsize=7, color='gray')
ax.set_xlabel('발산 (first-last distance)'); ax.set_ylabel('수렴 기울기')
ax.set_title('1-10(c) 발산×수렴 4사분면')

# (d) Quadrant × FI/FD cross-tabulation heatmap
ax = fig.add_subplot(12, 4, 26)
part_df['quad'] = ''
part_df.loc[(part_df['first_last_distance'] >= div_med) & (part_df['divergence_slope'] >= conv_med), 'quad'] = '지속적 탐색형'
part_df.loc[(part_df['first_last_distance'] >= div_med) & (part_df['divergence_slope'] < conv_med), 'quad'] = '탐색 후 안착형'
part_df.loc[(part_df['first_last_distance'] < div_med) & (part_df['divergence_slope'] >= conv_med), 'quad'] = '점진적 발산형'
part_df.loc[(part_df['first_last_distance'] < div_med) & (part_df['divergence_slope'] < conv_med), 'quad'] = '안정적 정교화형'

ct = pd.crosstab(part_df['quad'], part_df['participant_group'])
quad_order = ['지속적 탐색형', '탐색 후 안착형', '점진적 발산형', '안정적 정교화형']
ct = ct.reindex(quad_order, fill_value=0)
sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar=False)
ax.set_title('1-10(d) 4유형 × FI/FD 교차표')

# Fisher exact test (2×4 → Chi-square or sum of 2×2)
try:
    from scipy.stats import chi2_contingency
    chi2, p_fisher, _, _ = chi2_contingency(ct.values)
    ax.text(0.5, -0.15, f'χ²={chi2:.2f}, p={p_fisher:.4f}',
            transform=ax.transAxes, ha='center', fontsize=8)
except:
    pass

# ── Remaining subplots: ideation_ratio ──
ax1 = fig.add_subplot(12, 4, 27)
boxplot_with_points(ax1, fi_part['ideation_ratio'], fd_part['ideation_ratio'],
                    'Ideation 비율', '1-추가(a) FI/FD별 Ideation 비율')
record_stat('ideation_ratio', fi_part['ideation_ratio'], fd_part['ideation_ratio'],
            part_df['geft_score'], part_df['ideation_ratio'])

ax2 = fig.add_subplot(12, 4, 28)
scatter_geft(ax2, part_df['geft_score'], part_df['ideation_ratio'],
             'Ideation 비율', '1-추가(b) GEFT × Ideation 비율')

plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(OUT_DIR + 'axis1_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("axis1_results.png saved.")

# ── Save stats ──
stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(OUT_DIR + 'axis1_stats.csv', index=False, encoding='utf-8-sig')
print("axis1_stats.csv saved.")
print(stats_df.to_string())
