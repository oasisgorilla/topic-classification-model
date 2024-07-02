import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
"""
sentence transformers로 연결하고, json으로 연결정보 내보내기
연결정보는 존재하는 모든 연결쌍임
"""
# 파일 경로 설정
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
chapter_embeddings = {filename: model.encode(text).tolist() for filename, text in chapters.items()}

# 유사도 임계값 설정 (예: 0.8)
similarity_threshold = 0.8

# 유사도 계산 함수 정의
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 챕터 파일명 리스트
chapter_keys = list(chapters.keys())
num_chapters = len(chapter_keys)

# 유사도 임계값 이상의 쌍을 저장할 리스트
edges = []
for i in range(num_chapters):
    for j in range(i + 1, num_chapters):
        similarity = cosine_similarity(chapter_embeddings[chapter_keys[i]], chapter_embeddings[chapter_keys[j]])
        if similarity >= similarity_threshold:
            edges.append({"source": chapter_keys[i], "target": chapter_keys[j], "similarity": similarity})

# 노드와 링크 데이터 생성
nodes = [{"id": key} for key in chapter_keys]
links = [{"source": edge["source"], "target": edge["target"], "value": edge["similarity"]} for edge in edges]

# 데이터 저장
graph_data = {"nodes": nodes, "links": links}
with open('graph_data.json', 'w') as f:
    json.dump(graph_data, f)
