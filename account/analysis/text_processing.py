from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from konlpy.tag import Okt
import re
from django.core.cache import cache

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
    title = re.sub(r'\b(TV|News|8뉴스|오대영 라이브)\b', '', title, flags=re.IGNORECASE)

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
    title = re.sub(r'/\s*모아보는 뉴스|굿모닝연예|뉴스딱|실시간 e뉴스|생생지구촌', '', title)

    #'-MBC 중계방송' 제거
    title = re.sub(r'-\s*MBC\s*중계방송', '', title, flags=re.IGNORECASE)

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
    # 제목 전처리
    cleaned_title = clean_title(title)
    stop_words = {'그', '저', '것', '수'}
    processed_title = ' '.join([
        word for word, tag in okt.pos(cleaned_title)
        if tag in ['Noun', 'Verb', 'Adjective'] and word not in stop_words
    ])

    # 스크립트 전처리
    script_text = ""
    if transcript:
        if isinstance(transcript, list):  # transcript가 리스트일 경우
            transcript_text = ' '.join([
                item.get('text', '') for item in transcript
                if isinstance(item, dict) and len(item.get('text', '')) > 2
            ])
        else:
            print(f"Unexpected transcript format: {type(transcript)}")
            transcript_text = ""

        # 형태소 분석
        script_text = ' '.join([
            word for word, tag in okt.pos(transcript_text)
            if tag in ['Noun', 'Verb', 'Adjective'] and word not in stop_words
        ])

    # 제목과 스크립트 결합
    return f"{processed_title} {script_text}".strip()

def get_top10_chart_based(videos):
    """
    TF-IDF와 조회수 정규화를 기반으로 상위 10개 영상을 선정하고,
    코사인 유사도로 중복 제거. 캐싱을 추가하여 성능 개선.
    """
    cache_key = "top10_chart"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data

    corpus = []
    views_list = []
    processed_videos = []

    for video in videos:
        combined_text = process_text(video.title, video.transcript or [])
        corpus.append(combined_text)
        views_list.append(video.views)

        # 원본 비디오에 cleaned_title 추가
        video.cleaned_title = clean_title(video.title)
        processed_videos.append(video)

    # TF-IDF 계산
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 조회수 정규화
    scaler = MinMaxScaler()
    normalized_views = scaler.fit_transform([[view] for view in views_list]).flatten()

    # TF-IDF 가중치 + 조회수 점수 계산
    weighted_scores = normalized_views + tfidf_matrix.sum(axis=1).A.flatten()

    # 상위 N개 선정
    initial_sorted_videos = sorted(
        zip(processed_videos, weighted_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 코사인 유사도를 사용한 중복 제거
    similarity_matrix = cosine_similarity(tfidf_matrix)
    selected_videos = []
    seen_indices = set()
    threshold = 0.7  # 중복 판단 기준 유사도

    for i, (video, score) in enumerate(initial_sorted_videos):
        if i in seen_indices:
            continue  # 이미 처리된 비디오 건너뜀

        # 현재 비디오를 선택
        selected_videos.append(video)

        # 해당 비디오와 유사한 비디오를 찾아서 중복 처리
        for j in range(len(processed_videos)):
            if similarity_matrix[i, j] > threshold:
                seen_indices.add(j)

        # 최대 10개까지만 유지
        if len(selected_videos) >= 10:
            break

    # 캐싱 저장
    cache.set(cache_key, selected_videos, timeout=3600)  # 1시간 동안 캐싱
    return selected_videos