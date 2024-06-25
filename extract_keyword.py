"""
TF-IDF를 사용하여 txt파일 여러 개에서 의미 있는 키워드 10가지를 추출
"""
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import nltk

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# 한국어 불용어 목록 정의
korean_stopwords = ['이', '그', '저', '의', '를', '에', '가', '은', '는', '을', '로', '과', '와', '한', '하다', '있다', '없다', '되다', '않다', '이다', '있습니다', '합니다']

# 함수 정의: 파일에서 텍스트를 읽어오기
def read_text_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"파일 {filename}을 찾을 수 없습니다.")
        return ""

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
    tokens = [token for token in tokens if token not in korean_stopwords]  # 불용어 제거
    return ' '.join(tokens)

# 파일 처리 루프
textArr = []
while True:
    pageNum = input('input page number(int only, or type "exit" to quit): ')
    if pageNum.lower() == "exit":
        text = '\n'.join(textArr)
        break

    try:
        pageNum = int(pageNum)
    except ValueError:
        print("올바른 숫자를 입력하세요.")
        continue

    filename = f'./output_pages/image_{pageNum}.txt'
    textStr = read_text_from_file(filename)
    textArr.append(textStr)
    
    if not textStr:
        continue

print(text)

# 텍스트 분석: 문장 단위로 분해
sentences = nltk.sent_tokenize(text)

# 문장들을 전처리하고 형태소 분석 및 불용어 제거 수행
processed_sentences = [tokenize_and_remove_stopwords(preprocess(sentence)) for sentence in sentences]

# 키워드 추출: TF-IDF를 사용한 키워드 추출
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_sentences)

# 단어와 그 중요도를 매핑
keywords = vectorizer.get_feature_names_out()
importance = X.toarray().sum(axis=0)
keyword_importance = dict(zip(keywords, importance))

# 중요도 기준으로 상위 10개 키워드 추출
top_keywords = Counter(keyword_importance).most_common(10)

# 출력
print("Top Keywords:")
for keyword, score in top_keywords:
    print(f"{keyword}: {score}")
