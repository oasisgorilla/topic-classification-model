from gensim.models import Word2Vec
from datasets import load_dataset

# Wikimedia Wikipedia 데이터셋 로드
print("Loading dataset...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split='train[:1%]')
print("Dataset loaded.")

# 텍스트 전처리 및 단어 추출
def preprocess_text(text):
    text = text.lower()
    text = text.replace('row access strobe', 'row_access_strobe')
    text = text.replace('revolutions per minute', 'rpm')
    return text.split()

texts = [example['text'] for example in dataset]
processed_texts = [preprocess_text(text) for text in texts]

# Word2Vec 모델 학습
model = Word2Vec(processed_texts, vector_size=100, window=5, min_count=1, workers=4)

# 학습된 단어 임베딩 사용 예시
word_pairs = [
    ("routines", "thread"),
    ("row_access_strobe", "requests"),
    ("row-major", "array"),
    ("rpm", "revolutions_per_minute")
]

for pair in word_pairs:
    word1, word2 = pair
    if word1 in model.wv.key_to_index and word2 in model.wv.key_to_index:
        similarity = model.wv.similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
    else:
        print(f"At least one of the words '{word1}' or '{word2}' not in vocabulary.")
