import re
import numpy as np
import logging
logger = logging.getLogger(__name__)


# 무거운 ML 의존성(sklearn / konlpy / torch / transformers)은 모듈 최상단이 아니라
# 실제로 사용하는 함수 내부에서 import 한다.
#  - Django 기동 시 불필요한 라이브러리 로딩을 피해 시작 속도를 높이고,
#  - clean_title 등 순수 함수가 무거운 의존성 없이 import/테스트 가능하도록 분리.

def clean_title(title):
    """
    뉴스 제목에서 /방송사이름만 제거하고 나머지 텍스트는 유지하는 함수
    """
    # 대괄호 안의 내용 제거 (예: [이슈])
    title = re.sub(r'\[.*?\]', '', title)

    # 슬래시(/)로 시작하는 날짜 제거 (예: /2024년 11월 26일)
    title = re.sub(r'/\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일', '', title)

    # 날짜 형식 제거 (예: 2024.12.05)
    title = re.sub(r'\d{4}\.\d{2}\.\d{2}', '', title)

    # 괄호와 그 안의 내용 제거 (예: (2024.11.27))
    title = re.sub(r'\(.*?\)', '', title)

    # 특정 단어 제거 (예: TV, News, 8뉴스, 오대영 라이브)
    title = re.sub(r'\b(TV|News|8뉴스|오대영 라이브|15시 뉴스|12시 뉴스)\b', '', title, flags=re.IGNORECASE)

    # '｜지금 이 뉴스' 또는 '| 지금 이 뉴스' 제거
    title = re.sub(r'[｜|]\s*지금 이 뉴스', '', title)

    # 특정 단어 제거 (예: TV)
    title = re.sub(r'\bTV\b', '', title, flags=re.IGNORECASE)

    # /방송사 이름 제거 (예: /YTN, /JTBC, /KBS)
    title = re.sub(r'/\s*(YTN|JTBC|KBS|MBC|연합뉴스TV|SBS)\s*', '', title, flags=re.IGNORECASE)

    # 해시태그 제거 (예: #뉴스다)
    title = re.sub(r'#\S+', '', title)

    # 다중 공백 제거 (괄호 제거 후 남을 수 있는 공백 처리)
    title = re.sub(r'\s+', ' ', title)

    # 제목 끝의 마침표 제거
    title = re.sub(r'\.\s*$', '', title)

    # '-' 뒤에 "뉴스투데이"와 날짜 제거
    title = re.sub(r'-\s*MBC\s*뉴스투데이\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일', '', title)
    title = re.sub(r'아침&', '', title, flags=re.IGNORECASE)

    # '/ 모아보는 뉴스' 제거
    title = re.sub(r'/\s*모아보는 뉴스|굿모닝연예|뉴스딱|실시간 e뉴스|생생지구촌|정치컨설팅|편상욱의 뉴스브리핑|스토브리그', '', title)

    #'-MBC 중계방송' 제거
    title = re.sub(r'-\s*MBC\s*중계방송', '', title, flags=re.IGNORECASE)

    title = re.sub(r'-\s*KBS', '', title, flags=re.IGNORECASE)

    title = re.sub(r'-\s*MBC', '', title, flags=re.IGNORECASE)

    # 'YYYY년 MM월 DD일'형식 날짜 제거
    title = re.sub(r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일', '', title)

    title = re.sub(r'-\s*MBC\s*뉴스', '', title, flags=re.IGNORECASE)

    # 6시 뉴스 | 12/22 - 제거
    title = re.sub(r'6시\s*뉴스\s*\|', '', title, flags=re.IGNORECASE)

    # 특정 단어 제거 (예: MBC 뉴스, TV, News, 6시 뉴스 등)
    title = re.sub(r'\b(MBC\s*뉴스|6시\s*뉴스|TV|News|8뉴스|오대영\s*라이브|뉴스룸)\b', '', title, flags=re.IGNORECASE)

    # 날짜 형식 제거 (예: 2024.12.05, 12/22, (일) 등)
    title = re.sub(r'\d{4}\.\d{2}\.\d{2}', '', title)
    title = re.sub(r'\d{1,2}/\d{1,2}', '', title)
    title = re.sub(r'\(\S+\)', '', title)  # 괄호 안의 내용 제거 (예: (일))

    # | 밀착카메라 2024 결산 제거
    title = re.sub(r'\|\s*밀착카메라\s*\d{4}\s*결산', '', title, flags=re.IGNORECASE)

    # | - 패턴 제거
    title = re.sub(r'\|\s*-\s*', '', title)

    # 앞뒤 공백 제거
    return title.strip()


def process_text(title, transcript=None):
    """
    제목과 스크립트를 전처리하여 결합한 텍스트 반환
    """
    from konlpy.tag import Okt
    okt = Okt()
    cleaned_title = clean_title(title)
    stop_words = {'그', '저', '것', '수'}
    processed_title = ' '.join([word for word, tag in okt.pos(cleaned_title) if
                                tag in ['Noun', 'Verb', 'Adjective'] and word not in stop_words])

    script_text = ""
    if transcript:
        if isinstance(transcript, list):
            transcript_text = ' '.join([item.get('text', '') for item in transcript if
                                        isinstance(item, dict) and len(item.get('text', '')) > 2])
        else:
            logger.info(f"Unexpected transcript format: {type(transcript)}")
            transcript_text = ""
        script_text = ' '.join([word for word, tag in okt.pos(transcript_text) if
                                tag in ['Noun', 'Verb', 'Adjective'] and word not in stop_words])

    return f"{processed_title} {script_text}".strip()

def get_bert_similarity_batch(corpus, batch_size=32):
    """
       KoBERT를 사용하여 문서 간 유사도를 계산하는 함수
       """
    from kobert_transformers import get_tokenizer
    from transformers import BertModel
    import torch
    from sklearn.metrics.pairwise import cosine_similarity

    tokenizer = get_tokenizer()
    model = BertModel.from_pretrained('skt/kobert-base-v1')  # KoBERT 모델 로드
    model.eval()

    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            sentence_embedding = outputs.pooler_output  # [CLS] 토큰 임베딩 사용
        embeddings.append(sentence_embedding.numpy())

    # 모든 배치 결합
    try:
        embeddings = np.vstack(embeddings)
    except ValueError as e:
        raise ValueError(f"Error in vstack: {e}. Check batch processing.")

    # 코사인 유사도 계산
    similarity_matrix = cosine_similarity(embeddings)

    logger.info(f"Generated KoBERT Similarity Matrix Shape: {similarity_matrix.shape}")  # 디버깅 출력
    return similarity_matrix


def get_hybrid_similarity(tfidf_matrix, bert_similarity_matrix, weight_tfidf=0.5, weight_bert=0.5):
    """
    TF-IDF와 BERT 기반 유사도를 결합하여 최종 유사도 매트릭스 생성
    """
    combined_similarity = weight_tfidf * tfidf_matrix + weight_bert * bert_similarity_matrix
    return combined_similarity

def extract_keywords(corpus, n_keywords=5):
    """
    TF-IDF를 사용하여 각 문서에서 중요한 키워드를 추출하는 함수

    Parameters:
    corpus (list): 문서 텍스트 리스트
    n_keywords (int): 추출할 키워드 수

    Returns:
    dict: 각 문서별 중요한 키워드와 점수
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from konlpy.tag import Okt
    import numpy as np

    # 형태소 분석기 (Okt) 초기화
    okt = Okt()

    # 텍스트 전처리 함수
    def preprocess_text(text):
        # 명사만 추출
        nouns = okt.nouns(text)
        # 두 글자 이상의 명사만 선택
        nouns = [noun for noun in nouns if len(noun) > 1]
        return ' '.join(nouns)

    # 텍스트 전처리
    processed_corpus = [preprocess_text(text) for text in corpus]

    # TF-IDF 벡터라이저 설정
    tfidf = TfidfVectorizer(
        max_features=1000,  # 최대 1000개의 단어만 포함
        sublinear_tf=True,  # tf 값을 1 + log(tf)로 변환
        min_df=2  # 최소 두 개의 문서에서 등장한 단어만 포함
    )

    # TF-IDF 행렬 생성
    tfidf_matrix = tfidf.fit_transform(processed_corpus)

    # 단어 리스트 가져오기
    feature_names = np.array(tfidf.get_feature_names_out())

    # 각 문서별로 상위 키워드 추출
    keywords_by_doc = {}
    for idx, doc in enumerate(processed_corpus):
        # 문서별 TF-IDF 점수 추출
        tfidf_scores = tfidf_matrix[idx].toarray()[0]
        # 상위 n_keywords 단어의 인덱스 추출
        top_indices = tfidf_scores.argsort()[-n_keywords:][::-1]
        # 키워드와 점수 저장
        keywords = []
        for index in top_indices:
            keyword = feature_names[index]
            score = tfidf_scores[index]
            if score > 0:  # 점수가 0보다 큰 경우만 포함
                keywords.append({'keyword': keyword, 'score': float(score)})
        keywords_by_doc[idx] = keywords

    return keywords_by_doc


# 파일 마지막에 아래 함수 추가
def select_key_news(corpus, n_keywords=5, top_n=10):
    """
    TF-IDF 키워드 추출과 혼합 유사도를 사용해 주요 뉴스를 선정하는 함수.

    Parameters:
    corpus (list): 뉴스 텍스트 리스트
    n_keywords (int): 추출할 키워드 수
    top_n (int): 선정할 주요 뉴스 수

    Returns:
    list: 상위 주요 뉴스 텍스트와 키워드 리스트
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Step 1: TF-IDF 키워드 추출
    keywords_by_doc = extract_keywords(corpus, n_keywords=n_keywords)

    # Step 2: TF-IDF 및 BERT 혼합 유사도 계산
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(corpus)
    bert_similarity_matrix = get_bert_similarity_batch(corpus)
    hybrid_similarity_matrix = get_hybrid_similarity(
        cosine_similarity(tfidf_matrix),
        bert_similarity_matrix,
        weight_tfidf=0.4,
        weight_bert=0.6
    )

    # Step 3: 중요도 점수 계산 및 정렬
    relevance_scores = hybrid_similarity_matrix.sum(axis=1)  # 유사도 합산
    ranked_indices = np.argsort(relevance_scores)[::-1][:top_n]  # 상위 top_n 선정

    # Step 4: 결과 반환
    top_articles = [corpus[idx] for idx in ranked_indices]
    top_keywords = [keywords_by_doc[idx] for idx in ranked_indices]

    return top_articles, top_keywords


def get_top10_chart_based(videos, num_clusters=5, additional_news=3):
    """
    클러스터링, TF-IDF, BERT 코사인 유사도를 결합해 혼합 전략으로 상위 10개 동영상을 선정
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import MinMaxScaler

    # 가중치 설정
    TFIDF_WEIGHT = 0.4
    VIEW_WEIGHT = 0.4
    HYBRID_WEIGHT_TFIDF = 0.4
    HYBRID_WEIGHT_BERT = 0.6

    # 클러스터 수를 데이터 크기에 따라 동적으로 조정
    num_clusters = max(2, min(len(videos) // 5, 10))  # 최소 2개, 최대 10개 클러스터

    # 데이터 전처리
    corpus = []
    views_list = []
    processed_videos = []

    for video in videos:
        combined_text = process_text(video.title, video.transcript or [])
        corpus.append(combined_text)
        views_list.append(video.views)

        video.cleaned_title = clean_title(video.title)
        processed_videos.append(video)

    if not corpus:
        raise ValueError("Error: corpus is empty. No videos to process.")

    # TF-IDF 계산
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tfidf_similarity_matrix = cosine_similarity(tfidf_matrix)

    logger.info(f"TF-IDF Similarity Matrix Shape: {tfidf_similarity_matrix.shape}")  # 디버깅

    # BERT 계산
    bert_similarity_matrix = get_bert_similarity_batch(corpus, batch_size=32)

    # BERT 유사도 행렬 크기 확인
    logger.info(f"BERT Similarity Matrix Shape: {bert_similarity_matrix.shape}")  # 디버깅

    # Hybrid Similarity 계산
    if tfidf_similarity_matrix.shape != bert_similarity_matrix.shape:
        raise ValueError(f"Shape mismatch: TF-IDF {tfidf_similarity_matrix.shape}, BERT {bert_similarity_matrix.shape}")

    hybrid_similarity_matrix = get_hybrid_similarity(
        tfidf_similarity_matrix,  # TF-IDF 유사도 행렬
        bert_similarity_matrix,  # BERT 유사도 행렬
        HYBRID_WEIGHT_TFIDF,
        HYBRID_WEIGHT_BERT
    )

    logger.info(f"Hybrid Similarity Matrix Shape: {hybrid_similarity_matrix.shape}")  # 디버깅

    # Hybrid Similarity Matrix 정규화
    scaler = MinMaxScaler()
    normalized_hybrid_similarity = scaler.fit_transform(hybrid_similarity_matrix)

    # 클러스터링 수행 (K-Means)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_hybrid_similarity)

    # 클러스터별 대표 동영상 선택
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((i, processed_videos[i], views_list[i]))  # (인덱스, 동영상 객체, 조회수)

    cluster_representatives = []
    for cluster, videos in clusters.items():
        cluster_center = kmeans.cluster_centers_[cluster]
        if len(videos) > 0:
            best_video = min(
                videos,
                key=lambda x: (
                    0.6 * np.linalg.norm(normalized_hybrid_similarity[x[0]] - cluster_center)  # 유사도 중심성
                    - 0.4 * x[2]  # 조회수
                )
            )
            cluster_representatives.append(best_video[1])  # 동영상 객체 추가

    # 핵심 뉴스 추가 (조회수 기준)
    additional_videos_needed = max(0, 10 - len(cluster_representatives))
    remaining_videos = sorted(
        [video for video in processed_videos if video not in cluster_representatives],
        key=lambda x: x.views,
        reverse=True
    )
    additional_videos = remaining_videos[:additional_videos_needed]

    # 최종 10개 동영상 선정
    final_videos = cluster_representatives + additional_videos

    # 중복 제거 후 상위 10개
    final_videos = list({video.cleaned_title: video for video in final_videos}.values())
    final_videos = sorted(final_videos, key=lambda x: x.views, reverse=True)[:10]

    # 디버깅 로그
    logger.info("선택된 대표 동영상:")
    for video in final_videos:
        logger.info(f"- {video.cleaned_title} (조회수: {video.views})")

    return final_videos





