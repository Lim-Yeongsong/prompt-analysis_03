"""
Step 0-4: Preprocessing pipeline
- Data load, derived variables, morpheme analysis, keyword classification, participant aggregation
"""

import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

from scipy.spatial.distance import cosine
from scipy.stats import linregress
import umap

# ── 0. Seeds & Paths ──
np.random.seed(42)
DATA_PATH = '/home/user/prompt-analysis_03/project_260324/data/prompt_data_250324.csv'
EMB_PATH  = '/home/user/prompt-analysis_03/project_260324/embeddings.npy'
OUT_DIR   = '/home/user/prompt-analysis_03/project_260324/'

# ── 1. Load data ──
df = pd.read_csv(DATA_PATH)
embeddings = np.load(EMB_PATH)  # (120, 3072)
print(f"Data: {df.shape}, Embeddings: {embeddings.shape}")

# ── 1-1. Selection type processing ──
for idx, row in df[df['prompt_type'] == 'selection'].iterrows():
    ctx = str(row['prompt_context'])
    raw = str(row['prompt_raw']).strip()
    # If raw is just a number, extract that option from context
    if raw.isdigit():
        num = int(raw)
        # Try to find numbered options (1️⃣, 2️⃣, 3️⃣ or 1., 2., 3.)
        options = re.split(r'[1-9]️⃣|[1-9]\.\s', ctx)
        options = [o.strip() for o in options if o.strip()]
        if num <= len(options):
            df.at[idx, 'prompt_raw'] = options[num - 1].split('\n')[0].strip()
    elif raw in ('엉', '응', '네', '예', '좋아', '그래'):
        # Agreement - use the context as the chosen text (first meaningful line)
        lines = [l.strip() for l in ctx.split('\n') if l.strip()]
        df.at[idx, 'prompt_raw'] = ' '.join(lines[:3])

# Handle NaN prompt_raw (prompt_114)
df['prompt_raw'] = df['prompt_raw'].fillna('')
df['prompt_raw'] = df['prompt_raw'].replace('nan', '')

# ── 1-2. Derived variables ──
df['char_count'] = df['prompt_raw'].apply(len)
df['char_count_no_space'] = df['prompt_raw'].apply(lambda x: len(x.replace(' ', '')))

# prompt_combined (no image_caption column in data)
df['prompt_combined'] = '[프롬프트] ' + df['prompt_raw']

# ── 1-3. UMAP ──
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_coords = reducer.fit_transform(embeddings)
df['umap_x'] = umap_coords[:, 0]
df['umap_y'] = umap_coords[:, 1]

print("UMAP done.")

# ── 2. Korean Morpheme Analysis (Kkma) ──
print("Starting morpheme analysis with Kkma...")
from konlpy.tag import Kkma
kkma = Kkma()

# POS tag groups
NOUN_TAGS = {'NNG', 'NNP'}
VERB_TAGS = {'VV'}
ADJ_TAGS = {'VA'}
PARTICLE_TAGS = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC'}
ADVERB_TAGS = {'MAG', 'MAJ'}
CONTENT_TAGS = NOUN_TAGS | VERB_TAGS | ADJ_TAGS | ADVERB_TAGS
FUNCTION_TAGS = PARTICLE_TAGS | {'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'XSA'}
FOREIGN_TAGS = {'SL'}  # Foreign language

morph_results = []
for i, row in df.iterrows():
    text = str(row['prompt_raw'])
    if not text.strip():
        morph_results.append({
            'morpheme_count': 0, 'sentence_count': 1,
            'noun_count': 0, 'verb_count': 0, 'adjective_count': 0,
            'particle_count': 0, 'content_word_count': 0,
            'function_word_count': 0, 'unique_morpheme_count': 0,
            'foreign_word_count': 0, 'pos_tags': []
        })
        continue

    try:
        pos_tags = kkma.pos(text)
    except:
        pos_tags = []

    tags_only = [t for _, t in pos_tags]
    morphs_only = [m for m, _ in pos_tags]

    # Sentence count using regex
    sentences = re.split(r'[.?!。]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = max(len(sentences), 1)

    # Unique particles for diversity
    particle_morphs = [(m, t) for m, t in pos_tags if t in PARTICLE_TAGS]
    unique_particle_types = len(set(m for m, t in particle_morphs))
    total_particles = len(particle_morphs)

    morph_results.append({
        'morpheme_count': len(pos_tags),
        'sentence_count': sent_count,
        'noun_count': sum(1 for t in tags_only if t in NOUN_TAGS),
        'verb_count': sum(1 for t in tags_only if t in VERB_TAGS),
        'adjective_count': sum(1 for t in tags_only if t in ADJ_TAGS),
        'particle_count': total_particles,
        'content_word_count': sum(1 for t in tags_only if t in CONTENT_TAGS),
        'function_word_count': sum(1 for t in tags_only if t in FUNCTION_TAGS),
        'unique_morpheme_count': len(set(morphs_only)),
        'foreign_word_count': sum(1 for t in tags_only if t in FOREIGN_TAGS),
        'pos_tags': pos_tags,
        '_unique_particle_types': unique_particle_types,
        '_total_particles': total_particles,
    })
    if (i + 1) % 20 == 0:
        print(f"  Morpheme analysis: {i+1}/120")

morph_df = pd.DataFrame(morph_results)

# Derived morpheme metrics
morph_df['ttr'] = morph_df['unique_morpheme_count'] / morph_df['morpheme_count'].replace(0, np.nan)
morph_df['content_density'] = morph_df['content_word_count'] / morph_df['morpheme_count'].replace(0, np.nan)
morph_df['noun_ratio'] = morph_df['noun_count'] / morph_df['morpheme_count'].replace(0, np.nan)
morph_df['verb_ratio'] = morph_df['verb_count'] / morph_df['morpheme_count'].replace(0, np.nan)
morph_df['adj_ratio'] = morph_df['adjective_count'] / morph_df['morpheme_count'].replace(0, np.nan)
morph_df['avg_sentence_length'] = morph_df['morpheme_count'] / morph_df['sentence_count'].replace(0, np.nan)
morph_df['particle_diversity'] = morph_df['_unique_particle_types'] / morph_df['_total_particles'].replace(0, np.nan)
morph_df['foreign_ratio'] = morph_df['foreign_word_count'] / morph_df['morpheme_count'].replace(0, np.nan)

# Drop helper cols
morph_df = morph_df.drop(columns=['_unique_particle_types', '_total_particles'], errors='ignore')

# Merge
for col in morph_df.columns:
    if col != 'pos_tags':
        df[col] = morph_df[col].values

# Store pos_tags separately (not CSV-friendly)
pos_tags_list = morph_df['pos_tags'].tolist()

print("Morpheme analysis done.")

# ── 3. Context/Object Keyword Classification ──
# 맥락 키워드 (42개)
CONTEXT_KEYWORDS = [
    '학교', '캠퍼스', '교실', '강의실', '건물', '도서관', '기숙사',
    '화장실', '세면대', '복도', '로비', '엘리베이터', '계단',
    '출입구', '입구', '출구', '야외', '실내', '거리', '공간',
    '환경', '분위기', '조명', '배경', '상황', '장면', '일상', '생활',
    '이동', '동선', '대기', '혼잡', '방문', '수업', '공부', '충전',
    '사용', '이용', '편의', '불편', '문제', '개선'
]

# 객체 키워드 (47개)
OBJECT_KEYWORDS = [
    '멀티탭', '콘센트', '키오스크', '터치패드', '스크린', '디스플레이',
    '앱', '어플', '모바일', '휴대폰', '스마트폰', '비누', '디스펜서',
    '버튼', '패널', '표지판', '사인', '지도', '맵', '카메라', '센서',
    '레일', '천장', '벽걸이', '스탠드', '컵', '홀더', '아이콘',
    '마스코트', '캐릭터', '외형', '형태', '디자인', '제품', '기기',
    '장치', '시스템', '서비스', '플랫폼', '기능', '인터페이스', '화면',
    '스케치', '이미지', '목업', '렌더링', '도면'
]

print(f"Context keywords: {len(CONTEXT_KEYWORDS)}, Object keywords: {len(OBJECT_KEYWORDS)}")

def count_keywords(text, keywords):
    text = str(text)
    return sum(text.count(kw) for kw in keywords)

df['ctx_count'] = df['prompt_raw'].apply(lambda x: count_keywords(x, CONTEXT_KEYWORDS))
df['obj_count'] = df['prompt_raw'].apply(lambda x: count_keywords(x, OBJECT_KEYWORDS))
df['ctx_ratio'] = df['ctx_count'] / (df['ctx_count'] + df['obj_count']).replace(0, np.nan)

print("Keyword classification done.")

# ── 4. Participant-level aggregation ──
participants = []
for pid, grp in df.groupby('participant_id'):
    row0 = grp.iloc[0]
    emb_indices = grp.index.tolist()
    emb_sub = embeddings[emb_indices]

    # Turn count
    turn_count = len(grp)

    # Total char count
    total_char = grp['char_count'].sum()

    # Initial length
    initial = grp[grp['prompt_type'] == 'initial']
    initial_length = initial['char_count'].iloc[0] if len(initial) > 0 else grp['char_count'].iloc[0]

    # Mean refinement length
    refinements = grp[grp['prompt_type'] == 'refinement']
    mean_ref_length = refinements['char_count'].mean() if len(refinements) > 0 else np.nan

    # Total semantic change (sum of adjacent cosine distances)
    total_change = 0
    for i in range(1, len(emb_sub)):
        total_change += cosine(emb_sub[i-1], emb_sub[i])

    # First-last distance
    first_last_dist = cosine(emb_sub[0], emb_sub[-1]) if len(emb_sub) > 1 else 0

    # Reference image count
    ref_img_count = (grp['img_type'] == 'reference').sum()

    # Ideation ratio
    ideation_ratio = (grp['phase'] == 'ideation').mean()

    # Mean ctx_ratio
    mean_ctx_ratio = grp['ctx_ratio'].mean()

    # Divergence slope (linear regression of turn-by-turn cosine distances)
    if len(emb_sub) > 2:
        distances = [cosine(emb_sub[i-1], emb_sub[i]) for i in range(1, len(emb_sub))]
        x_turns = np.arange(len(distances))
        slope, _, _, _, _ = linregress(x_turns, distances)
    else:
        slope = 0

    # Mean morpheme metrics
    mean_morph = grp['morpheme_count'].mean()
    mean_ttr = grp['ttr'].mean()
    mean_cd = grp['content_density'].mean()
    mean_nr = grp['noun_ratio'].mean()
    mean_asl = grp['avg_sentence_length'].mean()
    mean_pd = grp['particle_diversity'].mean()

    participants.append({
        'participant_id': pid,
        'geft_score': row0['geft_score'],
        'participant_group': row0['participant_group'],
        'osivq_object_pct': row0['osivq_object_pct'],
        'osivq_spatial_pct': row0['osivq_spatial_pct'],
        'osivq_verbal_pct': row0['osivq_verbal_pct'],
        'osivq_cognitive_style': row0['osivq_cognitive_style'],
        'major': row0['major'],
        'workshop_group': row0['workshop_group'],
        'total_char_count': total_char,
        'initial_length': initial_length,
        'mean_refinement_length': mean_ref_length,
        'turn_count': turn_count,
        'total_change': total_change,
        'first_last_distance': first_last_dist,
        'ref_img_count': ref_img_count,
        'ideation_ratio': ideation_ratio,
        'mean_ctx_ratio': mean_ctx_ratio,
        'divergence_slope': slope,
        'mean_morpheme_count': mean_morph,
        'mean_ttr': mean_ttr,
        'mean_content_density': mean_cd,
        'mean_noun_ratio': mean_nr,
        'mean_avg_sentence_length': mean_asl,
        'mean_particle_diversity': mean_pd,
    })

part_df = pd.DataFrame(participants)
part_df = part_df.sort_values('participant_id').reset_index(drop=True)

# Save preprocessed data
df_save = df.drop(columns=['pos_tags'], errors='ignore')
df_save.to_csv(OUT_DIR + 'prompt_preprocessed.csv', index=False, encoding='utf-8-sig')
part_df.to_csv(OUT_DIR + 'participant_analysis.csv', index=False, encoding='utf-8-sig')

# Save as pickle for downstream scripts
import pickle
with open(OUT_DIR + 'preprocess_cache.pkl', 'wb') as f:
    pickle.dump({
        'df': df,
        'part_df': part_df,
        'embeddings': embeddings,
        'umap_coords': umap_coords,
        'pos_tags_list': pos_tags_list,
        'CONTEXT_KEYWORDS': CONTEXT_KEYWORDS,
        'OBJECT_KEYWORDS': OBJECT_KEYWORDS,
    }, f)

print(f"\nParticipant df: {part_df.shape}")
print(part_df[['participant_id', 'participant_group', 'geft_score', 'turn_count', 'total_char_count']].to_string())
print("\nPreprocessing complete! Files saved.")
