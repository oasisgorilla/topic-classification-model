import os
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 파일 경로 설정
# directory = 'C:/Users/chenj/Downloads/test2'
directory = './output_pages'

if not os.path.exists(directory):
    print(f"Error: Directory '{directory}' does not exist.")
    exit(1)

# 파일에서 텍스트 읽기
chapters = {}
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            chapters[filename] = file.read()


# Sentence-BERT 모델 로드
model = SentenceTransformer('all-mpnet-base-v2')

# 각 챕터를 임베딩으로 변환
chapter_embeddings = [model.encode(text) for text in chapters.values()]
chapter_embeddings_array = np.array(chapter_embeddings)

# 연관성 분석 (모든 챕터 쌍 간의 유사도 계산)
similarities = []

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

chapter_keys = list(chapters.keys())

# 유사도 임계값 설정 (예: 0.8)
similarity_threshold = 0.8

# 그룹 형성
groups = []
used = set()

for i, chapter1 in enumerate(chapter_keys):
    if chapter1 in used:
        continue
    group = [chapter1]
    for j, chapter2 in enumerate(chapter_keys):
        if i != j and chapter2 not in used:
            sim = cosine_similarity(chapter_embeddings[i], chapter_embeddings[j])
            if sim >= similarity_threshold:
                group.append(chapter2)
                used.add(chapter2)
    if len(group) > 1:
        groups.append(group)
    used.add(chapter1)

# 그룹 출력
print("형성된 그룹:")
for idx, group in enumerate(groups):
    print(f"그룹 {idx+1}: {group}")

# 그래프로 시각화
G = nx.Graph()

# 노드 추가
for group in groups:
    for chapter in group:
        G.add_node(chapter)

# 엣지 추가 (유사도 기준 상위 n개만 추가)
for group in groups:
    for i, chapter1 in enumerate(group):
        for j, chapter2 in enumerate(group):
            if i < j:
                sim = cosine_similarity(chapter_embeddings[chapter_keys.index(chapter1)], chapter_embeddings[chapter_keys.index(chapter2)])
                G.add_edge(chapter1, chapter2, weight=sim)

# 그래프 시각화
pos = nx.spring_layout(G)  # 레이아웃 설정
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', linewidths=1, font_size=15)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f'{v:.2f}' for k, v in labels.items()})
plt.title("챕터 간의 연관성 그래프")
plt.show()
