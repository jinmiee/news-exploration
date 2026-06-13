import re
import numpy as np
import logging
logger = logging.getLogger(__name__)


# 무거운 ML 의존성(sklearn / konlpy / torch / transformers)은 모듈 최상단이 아니라
# 실제로 사용하는 함수 내부에서 import 한다.
#  - Django 기동 시 불필요한 라이브러리 로딩을 피해 시작 속도를 높이고,
#  - clean_title 등 순수 함수가 무거운 의존성 없이 import/테스트 가능하도록 분리.

# 차트 중요도 점수 가중치 (합 1.0)
#  - 토픽 중심성: 여러 매체/영상이 동시에 다룬 큰 이슈일수록 높음(단순 조회수와 독립적인 '중요도' 신호)
#  - 조회수: 대중적 관심도(인기)
CHART_CENTRALITY_WEIGHT = 0.5
CHART_VIEW_WEIGHT = 0.5

# Okt 형태소 분석기는 JVM 초기화 비용이 크므로 프로세스당 1회만 생성해 재사용한다.
_OKT = None


def _get_okt():
    global _OKT
    if _OKT is None:
        from konlpy.tag import Okt
        _OKT = Okt()
    return _OKT


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
    okt = _get_okt()
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
    import numpy as np

    # 형태소 분석기 (Okt) — 싱글톤 재사용
    okt = _get_okt()

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


def get_top10_chart_based(videos, top_n=10):
    """
    유사 뉴스를 클러스터링해 토픽 중복을 제거하고,
    '토픽 중요도 + 조회수'를 결합한 중요도 점수로 Top N 을 선정한다.

    중요도 점수 = CHART_CENTRALITY_WEIGHT * 토픽중심성(정규화)
                + CHART_VIEW_WEIGHT     * 조회수(log 정규화)
      - 토픽중심성: 다른 기사들과의 (TF-IDF+KoBERT) 하이브리드 유사도 합.
        여러 매체/영상이 동시에 다룬 큰 이슈일수록 높음 → 조회수와 독립적인 '중요도' 신호.
      - 조회수: 분포가 크게 치우치므로 log 후 정규화.

    반환 리스트의 순서가 곧 최종 순위(rank)다. (호출부에서 enumerate 로 rank 부여)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans

    videos = list(videos)
    n = len(videos)
    if n == 0:
        raise ValueError("get_top10_chart_based: 입력 영상이 없습니다.")

    # 1) 전처리 + 조회수 수집
    corpus, views_list = [], []
    for v in videos:
        corpus.append(process_text(v.title, v.transcript or []))
        views_list.append(int(v.views or 0))
        v.cleaned_title = clean_title(v.title)

    # 표본이 너무 적으면 분석을 생략하고 중복 제거 + 조회수 정렬만 수행
    if n <= 2:
        uniq = list({v.cleaned_title: v for v in videos}.values())
        return sorted(uniq, key=lambda v: int(v.views or 0), reverse=True)[:top_n]

    # 2) TF-IDF + KoBERT 하이브리드 유사도 행렬
    tfidf_sim = cosine_similarity(TfidfVectorizer(max_features=5000).fit_transform(corpus))
    bert_sim = get_bert_similarity_batch(corpus, batch_size=32)
    if tfidf_sim.shape != bert_sim.shape:
        raise ValueError(f"Shape mismatch: TF-IDF {tfidf_sim.shape}, BERT {bert_sim.shape}")
    hybrid = get_hybrid_similarity(tfidf_sim, bert_sim, 0.4, 0.6)

    # 3) 중요도 구성요소를 같은 [0,1] 스케일로 정규화 후 결합
    mm = MinMaxScaler()
    centrality = hybrid.sum(axis=1)  # 토픽 중심성(다른 기사와의 유사도 총합)
    centrality_norm = mm.fit_transform(centrality.reshape(-1, 1)).ravel()
    views_norm = mm.fit_transform(np.log1p(np.array(views_list, dtype=float)).reshape(-1, 1)).ravel()
    importance = CHART_CENTRALITY_WEIGHT * centrality_norm + CHART_VIEW_WEIGHT * views_norm

    # 4) 유사 토픽 클러스터링 (중복 제거 / 다양성 확보)
    num_clusters = min(max(2, n // 5), 10, n)
    labels = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(hybrid)

    # 5) 클러스터별 대표 = 클러스터 내 중요도 최댓값 (스케일이 통일되어 정상 동작)
    selected = {}
    for i, label in enumerate(labels):
        if label not in selected or importance[i] > importance[selected[label]]:
            selected[label] = i
    selected_idx = set(selected.values())

    # 6) top_n 에 못 미치면 중요도 높은 순으로 채움
    for i in sorted(range(n), key=lambda k: importance[k], reverse=True):
        if len(selected_idx) >= top_n:
            break
        selected_idx.add(i)

    # 7) 중요도 내림차순 정렬 + cleaned_title 중복 제거 → 최종 순위
    final, seen = [], set()
    for i in sorted(selected_idx, key=lambda k: importance[k], reverse=True):
        title = videos[i].cleaned_title
        if title in seen:
            continue
        seen.add(title)
        videos[i].importance_score = round(float(importance[i]), 4)
        final.append(videos[i])
        if len(final) >= top_n:
            break

    logger.info("선택된 Top%d (중요도순):", len(final))
    for rank, v in enumerate(final, 1):
        logger.info("  %d. %s (조회수=%s, score=%.3f)", rank, v.cleaned_title, v.views, v.importance_score)

    return final





