import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from konlpy.tag import Okt
import re
from django.core.cache import cache
from fuzzywuzzy import fuzz
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch

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
            print(f"Unexpected transcript format: {type(transcript)}")
            transcript_text = ""
        script_text = ' '.join([word for word, tag in okt.pos(transcript_text) if
                                tag in ['Noun', 'Verb', 'Adjective'] and word not in stop_words])

    return f"{processed_title} {script_text}".strip()

def get_bert_similarity_batch(corpus, batch_size=32):
    """
    BERT를 사용하여 문장 임베딩을 생성하고 코사인 유사도 행렬을 반환
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    model.eval()

    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS 토큰의 출력 (Pooler Output)을 사용
            sentence_embedding = outputs.pooler_output
        embeddings.append(sentence_embedding.numpy())

    # Numpy 배열로 변환 후 코사인 유사도 계산
    import numpy as np
    embeddings = np.vstack(embeddings)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def get_hybrid_similarity(tfidf_matrix, bert_similarity_matrix, weight_tfidf=0.5, weight_bert=0.5):
    """
    TF-IDF와 BERT 기반 유사도를 결합하여 최종 유사도 매트릭스 생성
    """
    combined_similarity = weight_tfidf * tfidf_matrix + weight_bert * bert_similarity_matrix
    return combined_similarity

def get_top10_chart_based(videos, num_clusters=5, additional_news=3):
    """
    클러스터링, TF-IDF, BERT 코사인 유사도를 결합해 혼합 전략으로 상위 10개 동영상을 선정
    """
    # 가중치 설정
    TFIDF_WEIGHT = 0.4
    VIEW_WEIGHT = 0.4
    HYBRID_WEIGHT_TFIDF = 0.4
    HYBRID_WEIGHT_BERT = 0.6

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
        print("Error: corpus is empty. No videos to process.")
        return []

    # TF-IDF 계산
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # BERT 계산
    bert_similarity_matrix = get_bert_similarity_batch(corpus, batch_size=32)

    # Hybrid Similarity 계산
    hybrid_similarity_matrix = get_hybrid_similarity(
        tfidf_matrix,
        bert_similarity_matrix,
        HYBRID_WEIGHT_TFIDF,
        HYBRID_WEIGHT_BERT
    )

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
        best_video = min(
            videos,
            key=lambda x: (
                0.6 * np.linalg.norm(normalized_hybrid_similarity[x[0]] - cluster_center)  # 유사도 중심성
                - 0.4 * x[2]  # 조회수
            )
        )
        cluster_representatives.append(best_video[1])  # 동영상 객체 추가

    # 핵심 뉴스 추가 (조회수 기준)
    remaining_videos = [
        video for video in processed_videos if video not in cluster_representatives
    ]
    remaining_videos = sorted(remaining_videos, key=lambda x: x.views, reverse=True)

    # 부족한 개수만큼 조회수 기반 추가
    additional_videos = remaining_videos[:additional_news]

    # 최종 10개 동영상 선정
    final_videos = cluster_representatives + additional_videos
    final_videos = sorted(final_videos, key=lambda x: x.views, reverse=True)[:10]

    # 디버깅 로그
    print("선택된 대표 동영상:")
    for video in final_videos:
        print(f"- {video.cleaned_title} (조회수: {video.views})")

    return final_videos

