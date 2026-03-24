"""
통합 분석: FDR correction, correlation matrix, forest plot
Output: integrated_correlation.png, forest_plot.png, fdr_correction.csv
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
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

np.random.seed(42)

font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = '/home/user/prompt-analysis_03/project_260324/'

with open(OUT_DIR + 'preprocess_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
part_df = cache['part_df']
df = cache['df']

# Add extra columns if missing
for ecol in ['verb_ratio', 'adj_ratio', 'foreign_ratio']:
    pcol = f'mean_{ecol}'
    if pcol not in part_df.columns:
        part_df[pcol] = df.groupby('participant_id')[ecol].mean().reindex(part_df['participant_id']).values
if 'mean_sentence_count' not in part_df.columns:
    part_df['mean_sentence_count'] = df.groupby('participant_id')['sentence_count'].mean().reindex(part_df['participant_id']).values

# ═══════════════════════════════════════
# 1. FDR Correction across all axes
# ═══════════════════════════════════════
axis1 = pd.read_csv(OUT_DIR + 'axis1_stats.csv')
axis2 = pd.read_csv(OUT_DIR + 'axis2_stats.csv')
axis3 = pd.read_csv(OUT_DIR + 'axis3_stats.csv')

# Collect all p-values
fdr_rows = []

# Axis 1: 'p' column
for _, row in axis1.iterrows():
    fdr_rows.append({'axis': 'Axis1', '변인': row['변인'], 'test': 'Mann-Whitney U', 'p_original': row['p']})
    fdr_rows.append({'axis': 'Axis1', '변인': row['변인'] + ' (Spearman)', 'test': 'Spearman', 'p_original': row['Spearman p']})

# Axis 2: 'p' column
for _, row in axis2.iterrows():
    if pd.notna(row['p']):
        fdr_rows.append({'axis': 'Axis2', '변인': row['분석'], 'test': row['test'], 'p_original': row['p']})

# Axis 3: 'p' column
for _, row in axis3.iterrows():
    fdr_rows.append({'axis': 'Axis3', '변인': row['분석'], 'test': 'Mann-Whitney U', 'p_original': row['p']})
    if pd.notna(row.get('Spearman p', np.nan)):
        fdr_rows.append({'axis': 'Axis3', '변인': row['분석'] + ' (Spearman)', 'test': 'Spearman', 'p_original': row['Spearman p']})

fdr_df = pd.DataFrame(fdr_rows)
fdr_df = fdr_df.dropna(subset=['p_original'])

# Apply BH-FDR
reject, pvals_corr, _, _ = multipletests(fdr_df['p_original'].values, method='fdr_bh', alpha=0.10)
fdr_df['p_corrected'] = pvals_corr
fdr_df['significant_original'] = fdr_df['p_original'] < 0.05
fdr_df['significant_FDR'] = reject

fdr_df.to_csv(OUT_DIR + 'fdr_correction.csv', index=False, encoding='utf-8-sig')
print(f"FDR correction: {fdr_df['significant_original'].sum()} original sig, {fdr_df['significant_FDR'].sum()} after FDR")
print(fdr_df[fdr_df['significant_original']].to_string())

# ═══════════════════════════════════════
# 2. Integrated Correlation Matrix
# ═══════════════════════════════════════
corr_vars = {
    'geft_score': 'GEFT',
    'osivq_object_pct': 'OSIVQ Object',
    'osivq_spatial_pct': 'OSIVQ Spatial',
    'osivq_verbal_pct': 'OSIVQ Verbal',
    'total_char_count': '전체 글자수',
    'initial_length': '초기 길이',
    'mean_refinement_length': '수정 길이',
    'turn_count': '턴 수',
    'total_change': '의미 변화량',
    'first_last_distance': '첫-마지막 거리',
    'ref_img_count': '참조이미지',
    'mean_ctx_ratio': '맥락비율',
    'divergence_slope': '수렴기울기',
    'mean_morpheme_count': '형태소 수',
    'mean_ttr': 'TTR',
    'mean_content_density': '내용어밀도',
    'mean_noun_ratio': '명사비율',
    'mean_avg_sentence_length': '문장길이',
    'mean_particle_diversity': '조사다양성',
}

corr_cols = [c for c in corr_vars if c in part_df.columns]
corr_labels = [corr_vars[c] for c in corr_cols]

# Spearman correlation matrix
corr_data = part_df[corr_cols].copy()
n = len(corr_cols)
corr_matrix = np.zeros((n, n))
p_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        mask = corr_data.iloc[:, i].notna() & corr_data.iloc[:, j].notna()
        if mask.sum() > 2:
            r, p = spearmanr(corr_data.iloc[:, i][mask], corr_data.iloc[:, j][mask])
            corr_matrix[i, j] = r
            p_matrix[i, j] = p
        else:
            corr_matrix[i, j] = np.nan
            p_matrix[i, j] = 1.0

# Mask non-significant correlations
mask_sig = p_matrix > 0.05

fig, ax = plt.subplots(figsize=(16, 14))
fig.suptitle('통합 상관 행렬 (Spearman, p<0.05만 표시)', fontsize=16, fontweight='bold')

# Create masked array for annotation
annot = np.array([[f'{corr_matrix[i,j]:.2f}' if not mask_sig[i,j] else ''
                    for j in range(n)] for i in range(n)])

sns.heatmap(pd.DataFrame(corr_matrix, index=corr_labels, columns=corr_labels),
            annot=annot, fmt='', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            ax=ax, cbar_kws={'shrink': .8}, mask=mask_sig,
            xticklabels=True, yticklabels=True, linewidths=.5)
plt.setp(ax.get_xticklabels(), fontsize=8, rotation=45, ha='right')
plt.setp(ax.get_yticklabels(), fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR + 'integrated_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("integrated_correlation.png saved.")

# ═══════════════════════════════════════
# 3. Forest Plot (Effect Sizes)
# ═══════════════════════════════════════
# Collect all effect sizes and CIs
forest_data = []

# Axis 1
for _, row in axis1.iterrows():
    forest_data.append({
        'variable': row['변인'],
        'effect_type': 'rank-biserial r (FI vs FD)',
        'effect_size': row['rank-biserial r'],
        'ci_lo': row.get('Bootstrap 95% CI lower', np.nan),
        'ci_hi': row.get('Bootstrap 95% CI upper', np.nan),
        'p': row['p'],
        'axis': 'Axis 1',
    })
    forest_data.append({
        'variable': row['변인'] + ' (Spearman)',
        'effect_type': 'Spearman r (GEFT)',
        'effect_size': row['Spearman r (GEFT)'],
        'ci_lo': np.nan, 'ci_hi': np.nan,
        'p': row['Spearman p'],
        'axis': 'Axis 1',
    })

# Axis 3 (main MWU comparisons)
for _, row in axis3.iterrows():
    forest_data.append({
        'variable': row['분석'],
        'effect_type': 'rank-biserial r',
        'effect_size': row['rank-biserial r'],
        'ci_lo': row.get('CI_lo', np.nan), 'ci_hi': row.get('CI_hi', np.nan),
        'p': row['p'],
        'axis': 'Axis 3',
    })

fd = pd.DataFrame(forest_data)
fd = fd.dropna(subset=['effect_size'])
fd = fd.sort_values('effect_size')

fig, ax = plt.subplots(figsize=(12, max(8, len(fd) * 0.3)))
fig.suptitle('효과 크기 포레스트 플롯', fontsize=16, fontweight='bold')

y_pos = range(len(fd))
colors = []
for _, row in fd.iterrows():
    if row['p'] < 0.01:
        colors.append('#E53935')
    elif row['p'] < 0.05:
        colors.append('#FB8C00')
    elif row['p'] < 0.10:
        colors.append('#FDD835')
    else:
        colors.append('#78909C')

ax.barh(list(y_pos), fd['effect_size'], color=colors, alpha=0.7, height=0.6)
ax.axvline(0, color='black', ls='-', lw=0.8)

# CI error bars where available
for i, (_, row) in enumerate(fd.iterrows()):
    if pd.notna(row['ci_lo']) and pd.notna(row['ci_hi']):
        # Only plot if CI is in similar scale as effect size
        pass  # Skip CI bars for now as they're on different scales

ax.set_yticks(list(y_pos))
ax.set_yticklabels(fd['variable'], fontsize=7)
ax.set_xlabel('효과 크기 (rank-biserial r 또는 Spearman r)')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E53935', alpha=0.7, label='p < 0.01'),
    Patch(facecolor='#FB8C00', alpha=0.7, label='p < 0.05'),
    Patch(facecolor='#FDD835', alpha=0.7, label='p < 0.10'),
    Patch(facecolor='#78909C', alpha=0.7, label='p ≥ 0.10'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR + 'forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("forest_plot.png saved.")

print("\n=== All integrated analyses complete! ===")
