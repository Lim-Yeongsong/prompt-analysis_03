"""
Axis 3: Framework 분석 (2개 분석)
3-1: 디자인 프롬프트 분류 (Oppenlaender + 조유석)
3-2: 한국어 텍스트 복잡도 (Coh-Metrix inspired)
Output: axis3_results.png, axis3_stats.csv
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
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

np.random.seed(42)

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

# ════════════════════════════════════════════
# Analysis 3-1: Design Prompt Classification
# ════════════════════════════════════════════
# 6 categories keyword dictionaries (inductively built from prompt content)
CATEGORY_KEYWORDS = {
    '주제(Subject)': [
        '충전', '콘센트', '멀티탭', '비누', '디스펜서', '키오스크', '길찾기', '길안내',
        '캠퍼스', '맵', '지도', '표지판', '사인', '네비게이션', '안내', '정보',
        '학사', '수강', '시간표', '졸업', '건의', '엘리베이터', '대기', '동선',
        '책상', '컵', '홀더', 'QR', 'AR', '앱', '어플', '웹', '서비스', '플랫폼',
        '아이디어', '아이디에이션', '제품', '문제', '해결', '개선', '불편',
    ],
    '형태/스타일(Form/Style)': [
        '색상', '색깔', '컬러', '흰색', '하얀색', '회색', '파란', '빨간', '검정',
        '곡선', '직선', '둥근', '사각', '육면체', '원형', '타원', '미니멀', '모던',
        '심플', '깔끔', '트렌디', '감성', '세련', '클래식', '레트로', '미래',
        '스타일', '재질', '스테인리스', '나무', '금속', '플라스틱', '유리',
        '크기', '두께', '길이', '높이', '가로', '세로', '외형', '형태', '모양',
    ],
    '품질/기술(Quality/Technical)': [
        '렌더링', '목업', '스케치', '연필', '도면', '사실적', '현실적', '구체적',
        '사진', '이미지', '시각화', '해상도', '고화질', '디테일', '3D', '투시도',
        '도면', '실제', '목업', '카메라', '앵글', '구도', '비율',
    ],
    '맥락/환경(Context/Environment)': [
        '학교', '대학교', '홍익대', '교실', '강의실', '건물', '복도', '로비',
        '화장실', '세면대', '기숙사', '도서관', '야외', '실내', '입구', '출구',
        '배경', '환경', '공간', '장소', '설치', '배치', '벽', '천장', '바닥',
        '조명', '빛', '야간', '저녁', '인테리어', '분위기', '풍경',
    ],
    '기능/인터랙션(Function/Interaction)': [
        '터치', '누르', '당기', '인식', '센서', '자동', '수동', '무선', '충전',
        '연동', '연결', '이동', '회전', '접이', '올리', '내리', '잔량', '표시',
        '알림', '검색', '필터', '맞춤', '추천', '인터랙션', '버튼', '화면',
        '스크린', '디스플레이', '기능', 'GPS', 'LED', '레일', '롤링', '로보틱',
    ],
    '감성/분위기(Mood/Emotion)': [
        '편안', '편리', '편하', '귀여', '귀엽', '따뜻', '차가운', '시원',
        '깨끗', '위생', '안전', '친절', '불친절', '멋지', '예쁘', '아름다',
        '심미', '매력', '감성', '톤', '무드', '느낌', '현실감', '미래지향',
    ],
}

CATEGORY_NAMES = list(CATEGORY_KEYWORDS.keys())
CAT_SHORT = ['주제', '형태/스타일', '품질/기술', '맥락/환경', '기능/인터랙션', '감성/분위기']

# Count per prompt
for cat, kws in CATEGORY_KEYWORDS.items():
    col = f'cat_{cat}'
    df[col] = df['prompt_raw'].apply(lambda x: sum(str(x).count(kw) for kw in kws))

# Normalize to ratios per prompt
cat_cols = [f'cat_{c}' for c in CATEGORY_NAMES]
df['cat_total'] = df[cat_cols].sum(axis=1).replace(0, np.nan)
for col in cat_cols:
    df[col + '_ratio'] = df[col] / df['cat_total']

# Participant-level aggregation
for col in cat_cols:
    ratio_col = col + '_ratio'
    part_df[ratio_col] = df.groupby('participant_id')[ratio_col].mean().reindex(part_df['participant_id']).values

fi_part = part_df[part_df['participant_group'] == 'FI']
fd_part = part_df[part_df['participant_group'] == 'FD']

# ═══════ FIGURE ═══════
fig = plt.figure(figsize=(28, 48))
fig.suptitle('Axis 3: Framework 분석', fontsize=20, fontweight='bold', y=0.995)

# ── 3-1(a) FI/FD category distribution (grouped bar) ──
ax = fig.add_subplot(8, 3, 1)
x = np.arange(6)
fi_means = [fi_part[f'cat_{c}_ratio'].mean() for c in CATEGORY_NAMES]
fd_means = [fd_part[f'cat_{c}_ratio'].mean() for c in CATEGORY_NAMES]
fi_sems = [fi_part[f'cat_{c}_ratio'].sem()*1.96 for c in CATEGORY_NAMES]
fd_sems = [fd_part[f'cat_{c}_ratio'].sem()*1.96 for c in CATEGORY_NAMES]
ax.bar(x-.15, fi_means, .3, yerr=fi_sems, label='FI', color=FI_COLOR, alpha=.8, capsize=3)
ax.bar(x+.15, fd_means, .3, yerr=fd_sems, label='FD', color=FD_COLOR, alpha=.8, capsize=3)
ax.set_xticks(x); ax.set_xticklabels(CAT_SHORT, fontsize=7, rotation=15)
ax.legend(); ax.set_ylabel('비율'); ax.set_title('3-1(a) FI/FD별 카테고리 분포')

# Stats per category
cat_pvals = []
for cat in CATEGORY_NAMES:
    col = f'cat_{cat}_ratio'
    U, p, r = mwu(fi_part[col], fd_part[col])
    rs, ps = spearmanr(part_df['geft_score'], part_df[col].fillna(0))
    ci = bootstrap_ci(fi_part[col], fd_part[col])
    stats_rows.append({
        '분석': f'3-1 {cat}', 'test': 'Mann-Whitney U',
        'U': U, 'p': p, 'rank-biserial r': r,
        'CI_lo': ci[0], 'CI_hi': ci[1],
        'Spearman r': rs, 'Spearman p': ps,
        'FI mean': fi_part[col].mean(), 'FD mean': fd_part[col].mean(),
    })
    cat_pvals.append(p)

# FDR correction for 6 comparisons
reject, pvals_corr, _, _ = multipletests([p for p in cat_pvals if not np.isnan(p)], method='fdr_bh', alpha=0.1)
ax.text(0.02, 0.98, f'FDR corrected: {sum(reject)}/6 significant',
        transform=ax.transAxes, va='top', fontsize=7, family='monospace',
        bbox=dict(boxstyle='round', fc='wheat', alpha=.5))

# ── 3-1(b) GEFT × each category scatter (6) ──
for ci_idx, cat in enumerate(CATEGORY_NAMES):
    ax = fig.add_subplot(8, 3, 2 + ci_idx)
    col = f'cat_{cat}_ratio'
    ax.scatter(part_df['geft_score'], part_df[col], c='#333', s=25, alpha=.6)
    mask = part_df[col].notna()
    rs, ps = spearmanr(part_df.loc[mask, 'geft_score'], part_df.loc[mask, col])
    if mask.sum() > 2:
        z = np.polyfit(part_df.loc[mask, 'geft_score'], part_df.loc[mask, col], 1)
        xl = np.linspace(part_df['geft_score'].min(), part_df['geft_score'].max(), 50)
        ax.plot(xl, np.polyval(z, xl), 'k--', alpha=.5)
    sig = '***' if ps<.001 else '**' if ps<.01 else '*' if ps<.05 else 'ns'
    ax.set_xlabel('GEFT'); ax.set_ylabel('비율')
    ax.set_title(f'3-1(b) GEFT×{CAT_SHORT[ci_idx]}')
    ax.text(0.02, .98, f'r={rs:.3f} p={ps:.4f}{sig}', transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=.5))

# ── 3-1(c) Phase × category distribution ──
ax = fig.add_subplot(8, 3, 8)
x = np.arange(6)
w = 0.2
for pi, (phase, clr) in enumerate([('ideation', IDE_COLOR), ('final', FIN_COLOR)]):
    for gi, (grp, offset) in enumerate([('FI', -1), ('FD', 1)]):
        sub = df[(df['phase']==phase)&(df['participant_group']==grp)]
        vals = [sub[f'cat_{c}_ratio'].mean() for c in CATEGORY_NAMES]
        pos = x + (pi*2 + gi - 1.5)*w
        label = f'{grp}-{phase}'
        ax.bar(pos, vals, w*.9, label=label, color=clr, alpha=.6 if grp=='FD' else .9,
               edgecolor=FI_COLOR if grp=='FI' else FD_COLOR, linewidth=.5)
ax.set_xticks(x); ax.set_xticklabels(CAT_SHORT, fontsize=7, rotation=15)
ax.legend(fontsize=6, ncol=2); ax.set_ylabel('비율')
ax.set_title('3-1(c) Phase별 카테고리 분포 변화')

# ── 3-1(d) Participant category profile heatmap ──
ax = fig.add_subplot(8, 3, 9)
hm_data = part_df.set_index('participant_id')[[f'cat_{c}_ratio' for c in CATEGORY_NAMES]].copy()
hm_data.columns = CAT_SHORT
# Sort by group then GEFT
hm_data['group'] = part_df.set_index('participant_id')['participant_group']
hm_data = hm_data.sort_values(['group', 'participant_id'])
group_labels = hm_data['group']
hm_data = hm_data.drop(columns='group')
sns.heatmap(hm_data, cmap='YlOrRd', ax=ax, cbar_kws={'shrink': .6},
            yticklabels=True, xticklabels=True)
# Add group color labels
for i, (pid, grp) in enumerate(group_labels.items()):
    ax.get_yticklabels()[i].set_color(FI_COLOR if grp == 'FI' else FD_COLOR)
ax.set_title('3-1(d) 참여자별 카테고리 프로필 (파란=FI, 빨간=FD)')

# ════════════════════════════════════════════
# Analysis 3-2: Text Complexity (Coh-Metrix)
# ════════════════════════════════════════════
complexity_metrics = [
    ('morpheme_count', '총 형태소 수', 'mean_morpheme_count'),
    ('avg_sentence_length', '평균 문장 길이', 'mean_avg_sentence_length'),
    ('ttr', 'TTR (어휘 다양성)', 'mean_ttr'),
    ('content_density', '내용어 밀도', 'mean_content_density'),
    ('noun_ratio', '명사 비율', 'mean_noun_ratio'),
    ('particle_diversity', '조사 다양성', 'mean_particle_diversity'),
]

# Panel A: FI/FD boxplots (3×2)
for i, (prompt_col, label, part_col) in enumerate(complexity_metrics):
    ax = fig.add_subplot(8, 3, 10 + i)
    fi_v = fi_part[part_col]; fd_v = fd_part[part_col]
    data = [fi_v.dropna(), fd_v.dropna()]
    bp = ax.boxplot(data, labels=['FI', 'FD'], patch_artist=True)
    bp['boxes'][0].set_facecolor(FI_COLOR+'40'); bp['boxes'][1].set_facecolor(FD_COLOR+'40')
    for j, (vals, c) in enumerate(zip(data, [FI_COLOR, FD_COLOR])):
        ax.scatter(np.random.normal(j+1, .04, len(vals)), vals, c=c, s=20, alpha=.6)
    U, p, r = mwu(fi_v, fd_v)
    sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
    ax.text(0.02, .98, f'U={U:.0f} p={p:.4f}{sig}\nr={r:.3f}', transform=ax.transAxes, va='top',
            fontsize=7, family='monospace', bbox=dict(boxstyle='round', fc='wheat', alpha=.5))
    ax.set_ylabel(label); ax.set_title(f'3-2 A({chr(97+i)}) {label}')

# Panel B: GEFT scatter (3×2)
for i, (prompt_col, label, part_col) in enumerate(complexity_metrics):
    ax = fig.add_subplot(8, 3, 16 + i)
    geft = part_df['geft_score']; vals = part_df[part_col]
    mask = vals.notna()
    ax.scatter(geft[mask], vals[mask], c='#333', s=25, alpha=.6)
    rs, ps = spearmanr(geft[mask], vals[mask])
    if mask.sum() > 2:
        z = np.polyfit(geft[mask], vals[mask], 1)
        xl = np.linspace(geft.min(), geft.max(), 50)
        ax.plot(xl, np.polyval(z, xl), 'k--', alpha=.5)
    sig = '***' if ps<.001 else '**' if ps<.01 else '*' if ps<.05 else 'ns'
    ax.text(0.02, .98, f'r={rs:.3f} p={ps:.4f}{sig}', transform=ax.transAxes, va='top',
            fontsize=7, bbox=dict(boxstyle='round', fc='lightyellow', alpha=.5))
    ax.set_xlabel('GEFT'); ax.set_ylabel(label)
    ax.set_title(f'3-2 B({chr(97+i)}) GEFT×{label}')

# Stats for all 10 metrics (participant level)
all_complexity = [
    ('mean_morpheme_count', '총 형태소 수'),
    ('mean_avg_sentence_length', '평균 문장 길이'),
    ('mean_ttr', 'TTR'),
    ('mean_content_density', '내용어 밀도'),
    ('mean_noun_ratio', '명사 비율'),
    ('mean_particle_diversity', '조사 다양성'),
]
# Add extra metrics from prompt-level
extra_cols = {
    'verb_ratio': '동사 비율',
    'adj_ratio': '형용사 비율',
    'foreign_ratio': '외래어 비율',
}
for ecol, elabel in extra_cols.items():
    pcol = f'mean_{ecol}'
    if pcol not in part_df.columns:
        part_df[pcol] = df.groupby('participant_id')[ecol].mean().reindex(part_df['participant_id']).values
    all_complexity.append((pcol, elabel))

# sentence_count mean
if 'mean_sentence_count' not in part_df.columns:
    part_df['mean_sentence_count'] = df.groupby('participant_id')['sentence_count'].mean().reindex(part_df['participant_id']).values
all_complexity.append(('mean_sentence_count', '문장 수'))

fi_part = part_df[part_df['participant_group'] == 'FI']
fd_part = part_df[part_df['participant_group'] == 'FD']

complexity_pvals = []
for col, label in all_complexity:
    U, p, r = mwu(fi_part[col], fd_part[col])
    rs, ps = spearmanr(part_df['geft_score'], part_df[col].fillna(0))
    ci = bootstrap_ci(fi_part[col], fd_part[col])
    stats_rows.append({
        '분석': f'3-2 {label}', 'test': 'Mann-Whitney U',
        'U': U, 'p': p, 'rank-biserial r': r,
        'CI_lo': ci[0], 'CI_hi': ci[1],
        'Spearman r': rs, 'Spearman p': ps,
        'FI mean': fi_part[col].mean(), 'FD mean': fd_part[col].mean(),
    })
    complexity_pvals.append(p)

# FDR for 10 complexity comparisons
valid_p = [p for p in complexity_pvals if not np.isnan(p)]
if valid_p:
    reject2, pvals_corr2, _, _ = multipletests(valid_p, method='fdr_bh', alpha=0.1)

# Correlation heatmap of complexity metrics
ax = fig.add_subplot(8, 3, 22)
corr_cols = [c for c, _ in all_complexity if c in part_df.columns]
corr_labels = [l for c, l in all_complexity if c in part_df.columns]
corr_matrix = part_df[corr_cols].corr(method='spearman')
corr_matrix.columns = corr_labels
corr_matrix.index = corr_labels
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            cbar_kws={'shrink': .6}, xticklabels=True, yticklabels=True)
ax.set_title('3-2 텍스트 복잡도 지표 간 상관 행렬')
plt.setp(ax.get_xticklabels(), fontsize=6, rotation=45, ha='right')
plt.setp(ax.get_yticklabels(), fontsize=6)

plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(OUT_DIR + 'axis3_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("axis3_results.png saved.")

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(OUT_DIR + 'axis3_stats.csv', index=False, encoding='utf-8-sig')
print("axis3_stats.csv saved.")
print(stats_df.to_string())
