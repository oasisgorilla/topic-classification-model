import os  # os 모듈 임포트
import fitz  # PyMuPDF 라이브러리 임포트
import sqlite3  # SQLite 데이터베이스 라이브러리 임포트
from sklearn.feature_extraction.text import TfidfVectorizer  # 텍스트 벡터화를 위한 라이브러리 임포트
from sklearn.svm import SVC  # SVM 분류기를 위한 라이브러리 임포트

# PDF 파일에서 페이지별로 텍스트를 추출하는 함수
def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)  # PDF 문서를 엽니다
    pages = []  # 페이지 텍스트를 저장할 리스트
    for page_num in range(doc.page_count):  # 모든 페이지를 반복
        page = doc.load_page(page_num)  # 각 페이지를 로드
        pages.append(page.get_text())  # 페이지의 텍스트를 리스트에 추가
    return pages  # 페이지별 텍스트 리스트를 반환

# 예시 텍스트와 라벨
texts = ["Example text related to A", "Example text related to B", "Example text related to C"]
labels = ["A", "B", "C"]

# TF-IDF 벡터라이저를 사용하여 텍스트를 벡터화
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# SVM 모델을 학습
model = SVC()
model.fit(X, labels)

# 페이지 텍스트의 주제를 예측하는 함수
def predict_topic(page_text):
    X_test = vectorizer.transform([page_text])  # 테스트 데이터를 벡터화
    return model.predict(X_test)[0]  # 예측된 주제를 반환

# SQLite 데이터베이스를 초기화하는 함수
def init_db(db_dir='db', db_name='textbook.db'):
    # db 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    db_path = os.path.join(db_dir, db_name)  # 데이터베이스 파일 경로 생성
    conn = sqlite3.connect(db_path)  # 데이터베이스에 연결
    cursor = conn.cursor()  # 커서를 생성
    # 페이지 정보를 저장할 테이블 생성
    cursor.execute('''CREATE TABLE IF NOT EXISTS pages (
                        id INTEGER PRIMARY KEY,
                        page_num INTEGER,
                        text TEXT,
                        topic TEXT)''')
    conn.commit()  # 변경사항 커밋
    return conn  # 데이터베이스 연결 반환

# 페이지 정보를 데이터베이스에 저장하는 함수
def save_page_to_db(conn, page_num, text, topic):
    cursor = conn.cursor()  # 커서를 생성
    # 페이지 정보를 데이터베이스에 삽입
    cursor.execute("INSERT INTO pages (page_num, text, topic) VALUES (?, ?, ?)", 
                   (page_num, text, topic))
    conn.commit()  # 변경사항 커밋

# 특정 주제에 해당하는 페이지 번호를 가져오는 함수
def get_pages_by_topic(conn, topic):
    cursor = conn.cursor()  # 커서를 생성
    # 주제에 해당하는 페이지 번호를 선택
    cursor.execute("SELECT page_num FROM pages WHERE topic=?", (topic,))
    return cursor.fetchall()  # 결과 반환

pdf_path = './sample/Computer_Systems_A_Programmers_Perspective(3rd).pdf'  # PDF 파일 경로
pages = extract_text_by_page(pdf_path)  # PDF에서 페이지별로 텍스트를 추출

print(len(pages))

# db_conn = init_db()  # 데이터베이스 초기화

# # 각 페이지를 주제별로 분류하고 데이터베이스에 저장
# for page_num, page_text in enumerate(pages):
#     topic = predict_topic(page_text)  # 페이지 텍스트의 주제를 예측
#     save_page_to_db(db_conn, page_num, page_text, topic)  # 데이터베이스에 저장

# # 주제별로 페이지 목록을 가져옴
# topic_A_pages = get_pages_by_topic(db_conn, 'A')
# topic_B_pages = get_pages_by_topic(db_conn, 'B')
# topic_C_pages = get_pages_by_topic(db_conn, 'C')

# # 결과 출력
# print("Pages related to topic A:", topic_A_pages)
# print("Pages related to topic B:", topic_B_pages)
# print("Pages related to topic C:", topic_C_pages)
