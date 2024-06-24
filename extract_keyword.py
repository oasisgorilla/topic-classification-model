import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# 함수 정의: 파일에서 텍스트를 읽어오기
def read_text_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# 파일에서 텍스트 읽어오기
pageNum = input('input page number(int only) : ')
filename = f'./output_pages/page_{pageNum}.txt'  # 여기에 실제 파일 경로를 입력하세요
text = read_text_from_file(filename)

print(text)

# # 텍스트 분석: 문장 단위로 분해
# sentences = sent_tokenize(text)

# # 텍스트 전처리 및 정제 함수 정의
# def preprocess(text):
#     text = text.lower()  # 소문자로 변환
#     text = re.sub(r'\d+', '', text)  # 숫자 제거
#     text = re.sub(r'\s+', ' ', text)  # 다중 공백을 단일 공백으로 변환
#     text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
#     return text

# # 문장들을 전처리하여 리스트로 변환
# processed_sentences = [preprocess(sentence) for sentence in sentences]

# # 키워드 추출: TF-IDF를 사용한 키워드 추출
# vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
# X = vectorizer.fit_transform(processed_sentences)

# # 단어와 그 중요도를 매핑
# keywords = vectorizer.get_feature_names_out()
# importance = X.toarray().sum(axis=0)
# keyword_importance = dict(zip(keywords, importance))

# # 중요도 기준으로 상위 10개 키워드 추출
# top_keywords = Counter(keyword_importance).most_common(10)

# # 출력
# print("Top Keywords:")
# for keyword, score in top_keywords:
#     print(f"{keyword}: {score}")
