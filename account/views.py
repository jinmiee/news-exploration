'''
pip install konlpy networkx matplotlib pandas
'''
from collections import defaultdict
from datetime import timedelta, datetime

from django.http import JsonResponse
from django.utils import timezone

from django.utils.timezone import localtime, make_aware, is_aware
from django.views.decorators.csrf import csrf_exempt
import json
from django.contrib.auth.decorators import login_required
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer

from .forms import UserRegistrationForm
from .models import YouTubeData, Like, WeeklyIssue
from urllib.parse import urlparse, parse_qs

import pytz

import matplotlib
matplotlib.use('Agg')


from .analysis.relate_analysis import analyze_related_words, generate_network_graph
from .analysis.emotion_analysis import (
    generate_wordcloud,
    generate_pie_chart,
    analyze_morphemes,
    analyze_sentiment
)
from django.contrib.auth.models import User

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re

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
    title = re.sub(r'\b(MBC\s*뉴스|6시\s*뉴스|TV|News|8뉴스|오대영\s*라이브)\b', '', title, flags=re.IGNORECASE)

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
    코사인 유사도로 중복 제거.
    """
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
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
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

    return selected_videos

def chart(request):
    # 현재 시간 가져오기
    now = localtime()

    # 기준 시간 설정
    if now.hour < 11:  # 현재 시간이 오전 11시 이전
        analysis_start = (now - timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)
        analysis_end = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
    elif now.hour < 23:  # 현재 시간이 오전 11시 이후, 오늘 오후 11시 이전
        analysis_start = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
        analysis_end = now.replace(hour=11, minute=0, second=0, microsecond=0)
    else:  # 현재 시간이 오후 11시 이후
        analysis_start = now.replace(hour=11, minute=0, second=0, microsecond=0)
        analysis_end = now.replace(hour=23, minute=0, second=0, microsecond=0)

    # 데이터베이스에서 분석 시간에 해당하는 데이터 가져오기
    all_data = YouTubeData.objects.filter(
        upload_date__gte=analysis_start,
        upload_date__lt=analysis_end
    ).order_by('-views')

    # 상위 10개 데이터 선정 (중복 제거 포함)
    top_titles = get_top10_chart_based(all_data)
    for item in top_titles:
        item.id = str(item._id)  # 직접 _id 값을 id로 할당
    
    context = {
        "top_news": top_titles,
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "section": "chart"
    }
    return render(request, 'analysis/chart.html', context)

# 동영상 세부 정보 조회 API
@csrf_exempt # CSRF 검사 비활성화(POST 요청 허용)
def video_details(request):
    if request.method == 'POST':
        data = json.loads(request.body) # 요청 본문에서 JSON 데이터 로드
        video_url = data.get('url') # JSON 데이터에서 URL 추출
        video_title = data.get('title') # JSON 데이터에서 목 추출
        video_id = parse_qs(urlparse(video_url).query)['v'][0] # 동영상 ID 추출

        # 데이터베이스에서 URL로 YouTubeData 검색
        video = YouTubeData.objects.filter(url=video_url).first()

        # 검색된 데이터가 없을 경우 에러 응답
        if not video:
            return JsonResponse({"error": "Video not found."}, status=404)

        # 검색된 데이터가 있을 경우 URL 정보를 JSON 응답으로 반환
        return JsonResponse({"video_url": video_url, 'video_title': video_title, 'video_id': video_id})

    # 요청이 POST 방식이 아닐 경우 에러 응답
    return JsonResponse({"error": "Invalid request method."}, status=400)

def save_all_historical_top10():
    try:
        all_videos = YouTubeData.objects.all().order_by('upload_date')
        grouped_videos = defaultdict(list)

        seoul_tz = timezone('Asia/Seoul')

        for video in all_videos:
            # Null 또는 None 체크
            if not video.upload_date:
                print(f"Video {video.title} has no upload_date. Skipping...")
                continue

            # 문자열인 경우 datetime으로 변환
            if isinstance(video.upload_date, str):
                try:
                    video.upload_date = datetime.fromisoformat(video.upload_date)
                except ValueError:
                    print(f"Invalid date format for video {video.title}. Skipping...")
                    continue

            # UTC → KST 변환 후 날짜별 그룹화
            date_key = video.upload_date.astimezone(seoul_tz).date()
            grouped_videos[date_key].append(video)

        # 상위 10개 선정
        for date_key, videos in grouped_videos.items():
            top_videos = get_top10_chart_based(videos)

            for video in top_videos:
                WeeklyIssue.objects.update_or_create(
                    _id=video._id,
                    defaults={
                        'title': video.title,
                        'channel_name': video.channel_name,
                        'views': video.views,
                        'upload_date': video.upload_date,  # 이미 UTC로 저장됨
                        'url': video.url,
                        'channel': video.channel,
                        'thumbnail': video.thumbnail,
                        'comments': video.comments or [],
                        'transcript': video.transcript or []
                    }
                )
        print("All historical top 10 videos saved successfully.")
    except Exception as e:
        print(f"save_all_historical_top10 failed: {e}")

import logging
logger = logging.getLogger(__name__)

def save_daily_top10():
    """
    어제 날짜 데이터를 기반으로 상위 10개 비디오를 선정하고 저장
    """
    print("save_daily_top10 함수가 호출되었습니다.")  # 로그 추가
    try:
        # 오늘과 어제 날짜를 KST(Asia/Seoul) 기준으로 가져오기
        seoul_tz = timezone('Asia/Seoul')
        today = datetime.now(seoul_tz).date()
        yesterday = today - timedelta(days=1)

        # 어제의 시작과 끝을 UTC로 변환
        start_date = datetime.combine(yesterday, datetime.min.time()).replace(tzinfo=seoul_tz).astimezone(pytz.UTC)
        end_date = datetime.combine(yesterday, datetime.max.time()).replace(tzinfo=seoul_tz).astimezone(pytz.UTC)

        # 어제 날짜 데이터 필터링
        daily_videos = YouTubeData.objects.filter(upload_date__gte=start_date, upload_date__lt=end_date)

        if not daily_videos.exists():
            logger.warning("No videos found for yesterday.")
            return

        # 상위 10개 선정 (중복 제거 포함)
        top_videos = get_top10_chart_based(daily_videos)

        # 데이터 저장
        for video in top_videos:
            WeeklyIssue.objects.update_or_create(
                _id=video._id,
                defaults={
                    'title': video.title,
                    'channel_name': video.channel_name,
                    'views': video.views,
                    'upload_date': video.upload_date,  # 이미 UTC로 저장됨
                    'url': video.url,
                    'channel': video.channel,
                    'thumbnail': video.thumbnail,
                    'comments': video.comments or [],
                    'transcript': video.transcript or []
                }
            )
        logger.info("Daily top 10 videos saved successfully.")
    except Exception as e:
        logger.error(f"save_daily_top10 failed: {e}")

from datetime import timedelta

def weekly_issues(request):
    date_str = request.GET.get('date')
    seoul_tz = pytz.timezone('Asia/Seoul')
    today = datetime.now(seoul_tz).date()
    yesterday = today - timedelta(days=1)  # 어제 날짜 계산

    # 날짜 처리
    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            target_date = yesterday  # 잘못된 날짜 입력 시 어제 날짜로 설정
    else:
        target_date = yesterday  # 기본값: 어제 날짜

    # target_date가 어제보다 크다면 강제로 어제 날짜로 설정
    if target_date > yesterday:
        target_date = yesterday

    # 검색한 날짜 기준으로 내림차순 최근 6일 범위 계산
    start_date = target_date - timedelta(days=5)
    end_date = target_date + timedelta(days=1)

    # UTC 변환
    start_date_utc = datetime.combine(start_date, datetime.min.time()).astimezone(pytz.UTC)
    end_date_utc = datetime.combine(end_date, datetime.min.time()).astimezone(pytz.UTC)

    # 데이터 필터링
    issues = WeeklyIssue.objects.filter(
        upload_date__gte=start_date_utc,
        upload_date__lt=end_date_utc
    ).order_by('-upload_date')

    # 날짜별로 그룹화
    grouped_issues = defaultdict(list)
    for issue in issues:
        issue_date = issue.upload_date.astimezone(seoul_tz).date()
        weekday = issue_date.strftime("%Y년 %m월 %d일 (%a)").replace("Mon", "월").replace("Tue", "화").replace("Wed", "수").replace("Thu", "목").replace("Fri", "금").replace("Sat", "토").replace("Sun", "일")
        grouped_issues[weekday].append({
            'rank': len(grouped_issues[weekday]) + 1,
            'title': clean_title(issue.title),
            'views': issue.views,
            'url': issue.url,
        })

    # 정렬 및 최대 6개만 유지
    sorted_issues = sorted(grouped_issues.items(), key=lambda x: x[0], reverse=True)[:6]

    context = {
        'sorted_issues': sorted_issues,
        'target_date': target_date,
        'yesterday': yesterday,  # 어제 날짜를 템플릿에 전달
    }

    return render(request, 'analysis/weekly_issues.html', context)

#상세분석
# @login_required
def detail(request):
    video_url = request.GET.get('url')
    video_id = request.GET.get('id')

    # 선택된 비디오 데이터 가져오기
    video = YouTubeData.objects.filter(url=video_url).first()
    video_views = video.views if video else None
    video_likes = video.likes if video else None
    video_comments = video.comments if video else None
    video_title = video.title if video else None

    # 관련 비디오 처리
    all_videos = YouTubeData.objects.all()

    def process_for_similarity(video_data):
        title = clean_title(video_data.title)
        transcript = " ".join([item['text'] for item in video_data.transcript]) if video_data.transcript else ""
        return f"{title} {transcript}"

    # TF-IDF 벡터화
    corpus = [process_for_similarity(v) for v in all_videos]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    target_index = list(all_videos).index(video)
    similarity_scores = cosine_similarity(tfidf_matrix[target_index:target_index+1], tfidf_matrix).flatten()

    # 유사도가 높은 비디오 필터링
    threshold = 0.7
    related_videos = [
        {
            "url": v.url,
            "video_id": v.url.split('v=')[1].split('&')[0],  # 여기서 ID를 추출
            "thumbnail": v.thumbnail,
            "title": v.title,
        }
        for i, v in enumerate(all_videos)
        if similarity_scores[i] > threshold and i != target_index
    ]

    context = {
        'video': video,
        'video_id': video_id,
        'video_url': video_url,
        'video_title': video_title,
        'video_views': video_views,
        'video_likes': video_likes,
        'video_comments': video_comments,
        'related_videos': related_videos,  # 유사한 비디오들
    }

    return render(request, 'analysis/detail.html', context)

from transformers import pipeline
from django.shortcuts import render
from konlpy.tag import Okt
from .models import YouTubeData

# 감정 분석 파이프라인 설정
sentiment_analysis_pipeline = pipeline("sentiment-analysis",truncation=True, padding=True, max_length=512)

def analyze_sentiment(comment_text):
    try:
        from .analysis.emotion_analysis import analyze_sentiment as analyze_single_sentiment
        return analyze_single_sentiment(comment_text)
    except Exception as e:
        print(f"감정 분석 오류: {e}")
        return {
            'comment': comment_text,
            'sentiment': 'POSITIVE',
            'confidence': 0.6
        }

def emotion(request):
    video_url = request.GET.get('url')
    print(f"요청된 비디오 URL: {video_url}")  # 디버깅 로그
    
    video = YouTubeData.objects.filter(url=video_url).first()
    
    if video and video.comments:
        try:
            print(f"댓글 데이터 타입: {type(video.comments)}")  # 디버깅 로그
            print(f"첫 번째 댓글 샘플: {video.comments[0] if video.comments else 'No comments'}")  # 디버깅 로그
            
            # 댓글 데이터 추출 및 유효성 검사
            video_comments = []
            for comment in video.comments[:100]:
                if isinstance(comment, dict):
                    comment_text = comment.get('comment', '')
                elif isinstance(comment, str):
                    comment_text = comment
                else:
                    comment_text = str(comment)
                
                if comment_text.strip():  # 빈 댓글 제외
                    video_comments.append(comment_text)
            
            print(f"처리할 댓글 수: {len(video_comments)}")  # 디버깅 로그
            
            if not video_comments:
                raise ValueError("유효한 댓글이 없습니다.")
            
            # 감정 분석
            analyzed_comments = []
            for comment in video_comments:
                try:
                    result = analyze_sentiment(comment)
                    analyzed_comments.append(result)
                except Exception as e:
                    print(f"개별 댓글 분석 오류: {e}")  # 디버깅 로그
                    continue
            
            if not analyzed_comments:
                raise ValueError("감정 분석 결과가 없습니다.")
            
            context = {
                'section': 'emotion',
                'video_title': video.title,
                'video_id': request.GET.get('id'),  # video.video_id 대신 요청에서 가져옴
                'analyzed_comments': analyzed_comments[:10],
            }
            
            # 워드클라우드 생성
            try:
                wordcloud_image = generate_wordcloud(video_comments, analyzed_comments)
                if wordcloud_image:
                    context['wordcloud_image'] = wordcloud_image
            except Exception as e:
                print(f"워드클라우드 생성 오류: {e}")  # 디버깅 로그
            
            # 파이차트 생성
            try:
                pie_chart_image = generate_pie_chart(analyzed_comments)
                if pie_chart_image:
                    context['pie_chart_image'] = pie_chart_image
            except Exception as e:
                print(f"파이차트 생성 오류: {e}")  # 디버깅 로그
            
            # 형태소 분석
            try:
                rank_table = analyze_morphemes(analyzed_comments)
                if rank_table:
                    context['rank_table_by_morpheme'] = rank_table
            except Exception as e:
                print(f"형태소 분석 오류: {e}")  # 디버깅 로그
            
            return render(request, 'analysis/emotion.html', context)
            
        except Exception as e:
            print(f"전체 처리 오류: {str(e)}")  # 디버깅 로그
            context = {
                'section': 'emotion',
                'error_message': f'분석 처리 중 오류가 발생했습니다: {str(e)}'
            }
    else:
        context = {
            'section': 'emotion',
            'error_message': '비디오를 찾을 수 없거나 댓글이 없습니다.'
        }
    
    return render(request, 'analysis/emotion.html', context)

from django.db.models import Q
from functools import reduce
from operator import or_

def relate(request):
    video_url = request.GET.get('url')
    video_id = request.GET.get('id')
    
    video = YouTubeData.objects.filter(url=video_url).first()
    
    if video and video.transcript:
        try:
            # 제목 불용어 처리
            cleaned_title = clean_title(video.title)
            
            # transcript 데이터를 시간 단위로 구분하여 텍스트로 변환
            transcript_text = ' '.join([item['text'] for item in video.transcript])
            # 시간별 자막 데이터 생성
            transcript_segments = []
            for item in video.transcript:
                if 'start' in item and 'text' in item:
                    start_time = int(float(item['start']))
                    minutes = start_time // 60
                    seconds = start_time % 60
                    time_str = f"{minutes:02d}:{seconds:02d}"
                    transcript_segments.append({
                        'time': time_str,
                        'text': item['text']
                    })
            
            graph, top_pairs, important_keywords = analyze_related_words(transcript_text)
            network_graph = generate_network_graph(graph)
            
            # 키워드별 관련 뉴스 분류
            categorized_news = {}
            if important_keywords:
                for keyword in important_keywords:
                    related_news = YouTubeData.objects.filter(
                        title__icontains=keyword
                    ).exclude(url=video_url)[:6]
                    
                    if related_news:
                        cleaned_news = []
                        for news in related_news:
                            news.title = clean_title(news.title)
                            try:
                                news.video_id = news.url.split('v=')[1].split('&')[0]
                            except:
                                news.video_id = None
                            cleaned_news.append(news)
                        categorized_news[keyword] = cleaned_news
            
            context = {
                'section': 'relate',
                'video': video,
                'video_title': cleaned_title,
                'network_graph': network_graph,
                'top_pairs': top_pairs,
                'categorized_news': categorized_news,
                'important_keywords': important_keywords,
                'transcript_text': transcript_text,
                'transcript_segments': transcript_segments  # 시간별 자막 데이터 추가
            }
        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            context = {
                'section': 'relate',
                'error_message': '분석 중 오류가 발생했습니다.'
            }
    else:
        context = {
            'section': 'relate',
            'error_message': '비디오 자막이 없거나 비디오를 찾을 수 없습니다.'
        }
    
    return render(request, 'analysis/relate.html', context)

@login_required
def mypage(request):
    return render(request, 'analysis/mypage/mypage.html', {'section': 'mypage'}) 


@login_required
def like_video(request, video_id):
    try:
        video = YouTubeData.objects.get(_id=ObjectId(video_id))
        like_obj, created = Like.objects.get_or_create(
            user=request.user,
            youtube_data=video
        )
        
        if not created:  # 이미 좋아요가 있으면 삭제
            like_obj.delete()
            is_liked = False
        else:  # 새로 생성된 경우
            is_liked = True
            
        return JsonResponse({
            'status': 'success',
            'is_liked': is_liked
        })
    except Exception as e:
        print(f"Error: {str(e)}")  # 디버깅용
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

# @login_required
def my_liked_videos(request):
    liked_videos = YouTubeData.objects.filter(like__user=request.user)  
    
    context = {
        'liked_videos': liked_videos,
        'section': 'mypage'
    } 
    for video in liked_videos:
        print(video)
        video.id = str(video._id)
        print(video.id)
    return render(request, 'analysis/mypage/my_liked_videos.html', context)




    


def register(request):
    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        if user_form.is_valid():
            new_user = user_form.save(commit=False)
            new_user.set_password(user_form.cleaned_data['password'])
            new_user.save()
            return render(request, 'registration/register_done.html', {'new_user': new_user})
    else:
        user_form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'user_form': user_form})


def find_username(request):
    username = None
    error = None
    
    if request.method == "POST":
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)
            username = user.username
        except User.DoesNotExist:
            error = "사용자를 찾을 수 없습니다."
    
    return render(request, 'registration/find_username.html', {"username": username, "error": error})

from django.core.mail import send_mail

def send_password_reset_email(user, temp_password):
    subject = "비밀번호 초기화 안내"
    message = f"안녕하세요 {user.username}님,\n\n초기화된 임시 비밀번호는 다음과 같습니다: {temp_password}\n로그인 후 비밀번호를 변경해주세요."
    from_email = "namsugb99@gmail.com"
    recipient_list = [user.email]
    send_mail(subject, message, from_email, recipient_list)


def find_password(request):
    error = None
    if request.method == "POST":
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)
            # 비밀번호 초기화 이메일 전송
            temp_password = User.objects.make_random_password()
            user.set_password(temp_password)
            user.save()
            # 비밀번호 초기화 이메일 전송
            send_password_reset_email(user, temp_password)
            return render(request, 'registration/find_password_done.html', {'email': email})
        except User.DoesNotExist:
            error = "사용자를 찾을 수 없습니다."
    
    return render(request, 'registration/find_password.html', {"error": error})



