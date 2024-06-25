import os
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import nltk

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from nltk.corpus import stopwords

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
# 영어 불용어
stop_words = set(stopwords.words('english'))

# 한국어 불용어 목록 정의
korean_stopwords = ['이', '그', '저', '의', '를', '에', '가', '은', '는', '을', '로', '과', '와', '한', '하다', '있다', '없다', '되다', '않다', '이다', '있습니다', '합니다']

# 텍스트 전처리 및 정제 함수 정의
def preprocess(text):
    text = text.lower()  # 소문자로 변환
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = re.sub(r'\s+', ' ', text)  # 다중 공백을 단일 공백으로 변환
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    return text

# 형태소 분석 및 불용어 제거 함수 정의
def tokenize_and_remove_stopwords(text):
    okt = Okt()
    tokens = okt.morphs(text)  # 형태소 분석
    tokens = [token for token in tokens if token not in stop_words]  # 불용어 제거
    return ' '.join(tokens)

# 디렉토리 내부의 모든 txt 파일을 불러옴
def load_and_preprocess_files(directory):
    texts = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                preprocessed_text = preprocess(text)
                texts.append(preprocessed_text)
                filenames.append(filename)
    return texts, filenames

# TF-IDF를 사용하여 특징 추출
def extract_features(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix

# 코사인 유사도를 사용한 연관성 분석
def calculate_similarity(tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

# 시각화 및 클러스터링
def plot_similarity_matrix(similarity_matrix, filenames):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=filenames, yticklabels=filenames, annot=True, cmap='coolwarm')
    plt.title('Document Similarity Matrix')
    plt.show()

def cluster_documents(tfidf_matrix, num_clusters=20):
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(tfidf_matrix)
    return km.labels_

def print_clustered_documents(filenames, labels):
    clusters = {}
    for filename, label in zip(filenames, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)
    
    for cluster, docs in clusters.items():
        print(f"Cluster {cluster}: {docs}")

# 실행부
directory = './output_pages'

# 텍스트 파일 로드 및 전처리
texts, filenames = load_and_preprocess_files(directory)

# 특징 추출
tfidf_matrix = extract_features(texts)

# 유사도 계산
similarity_matrix = calculate_similarity(tfidf_matrix)

# 유사도 행렬 시각화
plot_similarity_matrix(similarity_matrix, filenames)

# 문서 클러스터링
labels = cluster_documents(tfidf_matrix)
print_clustered_documents(filenames, labels)

