'''
pip install konlpy networkx matplotlib pandas
'''
from collections import defaultdict
from datetime import timedelta, datetime

from django.db.models.functions import TruncDate
from django.http import JsonResponse
from django.utils import timezone
from django.shortcuts import render, get_object_or_404, redirect
from django.utils.timezone import localtime, make_aware, is_aware
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import date
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseBadRequest
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer

from .forms import UserRegistrationForm
from .models import YouTubeData, Like, WeeklyIssue
from urllib.parse import urlparse, parse_qs
import re
import pytz
from dateutil.parser import parse

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from transformers import pipeline

from konlpy.tag import Okt
from collections import Counter
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager

from .analysis.relate_analysis import analyze_related_words, generate_network_graph
from .analysis.emotion_analysis import (
    generate_wordcloud,
    generate_pie_chart,
    analyze_morphemes,
    analyze_sentiment
)

from .analysis.clustering import choose_10


from django.contrib.auth.models import User

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re

def process_titles_and_scripts(request):
    '''
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

    # 데이터 가져오기 (기준 시간에 맞는 데이터 필터링)
    all_data = YouTubeData.objects.filter(upload_date__gte=analysis_start, upload_date__lt=analysis_end).order_by('-views')
    '''
    # 초기화 및 전처리 설정
    okt = Okt()
    UNNECESSARY_TAGS = [
        'Josa', 'Conj', 'Punctuation', 'Eomi', 'Suffix', 'Foreign',
        'KoreanParticle', 'Alpha', 'Exclamation'
    ]
    start_date = datetime(2024, 12, 4)
    end_date = start_date + timedelta(days=1)
    all_data = YouTubeData.objects.filter(upload_date__gte=start_date, upload_date__lt=end_date).order_by('-views')

    processed_titles = []
    corpus = []
    views_list = []

    # 텍스트 전처리 함수
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)  # 여러 공백을 단일 공백으로 변환
        text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
        return text.strip()

    for data in all_data:
        # 제목 전처리
        try:
            cleaned_title = clean_text(
                ' '.join([word for word, tag in okt.pos(data.title) if tag not in UNNECESSARY_TAGS]))
        except Exception as e:
            print(f"Title processing failed for: {data.title}, Error: {e}")
            cleaned_title = "제목 없음"

        # 스크립트 전처리
        if data.transcript and isinstance(data.transcript, list):
            script_text = ' '.join(
                [item.text for item in data.transcript if hasattr(item, 'text') and isinstance(item.text, str)])
            cleaned_script = clean_text(
                ' '.join([word for word, tag in okt.pos(script_text) if tag not in UNNECESSARY_TAGS]))
        else:
            cleaned_script = "스크립트 없음"

        # 제목과 스크립트 결합
        combined_text = f"{cleaned_title} {cleaned_script}"
        print(f"Processed Combined Text: {combined_text[:100]}")  # 디버깅용 출력

        # 데이터 저장
        corpus.append(combined_text)
        views_list.append(data.views)

        processed_titles.append({
            "original_title": data.title,
            "cleaned_title": cleaned_title,
            "cleaned_script": cleaned_script,
            "combined_text": combined_text,
            "upload_date": data.upload_date,
            "channel": data.channel_name,
            "url": data.url,
            "views": data.views
        })

    # TF-IDF 분석
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 조회수 정규화 및 가중치 계산
    scaler = MinMaxScaler()
    normalized_views = scaler.fit_transform([[view] for view in views_list]).flatten()
    weighted_scores = normalized_views + tfidf_matrix.sum(axis=1).A.flatten()

    # 코사인 유사도 계산
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 가중치를 기준으로 정렬
    sorted_titles = sorted(
        processed_titles,
        key=lambda x: weighted_scores[processed_titles.index(x)],
        reverse=True
    )

    # 상위 10개 기사 선택
    chart_titles = sorted_titles[:10]

    # 중복된 기사 그룹화
    duplicates = []
    seen_titles = set()  # 중복 확인용

    for chart_title in chart_titles:
        duplicate_group = []
        for i, data in enumerate(processed_titles):
            # 차트의 기사와 동일하지 않으면서 유사도가 0.7 이상인 기사 찾기
            if data != chart_title and similarity_matrix[processed_titles.index(chart_title)][i] > 0.7:
                # 중복된 기사 중복 방지
                if data["original_title"] not in seen_titles:
                    duplicate_group.append(data)
                    seen_titles.add(data["original_title"])

        if duplicate_group:
            duplicates.append({
                "original": chart_title,
                "duplicates": duplicate_group
            })

    # 템플릿으로 전달
    context = {
        "processed_titles": chart_titles,  # 상위 10개 기사
        "duplicates": duplicates,         # 중복된 기사 목록
        "processing_stats": {
            "total_videos": len(all_data),
            "processed_titles": len(processed_titles),
            "unique_titles": len(chart_titles),
            "duplicate_groups": len(duplicates)
        },
        "section": "processed_data"
    }

    return render(request, 'analysis/processed_data.html', context)



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

    # '/ 모아보는 뉴스' 제거
    title = re.sub(r'/\s*모아보는 뉴스|굿모닝연예|뉴스딱', '', title)

    #'-MBC 중계방송' 제거
    title = re.sub(r'-\s*MBC\s*중계방송', '', title, flags=re.IGNORECASE)

    # 'YYYY년 MM월 DD일'형식 날짜 제거
    title = re.sub(r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일', '', title)

    # 앞뒤 공백 제거
    return title.strip()



def chart(request):
    now = localtime()  # 현재 시간 가져오기

    # 기준 시간 설정
    if now.hour < 11:  # 현재 시간이 오전 11시 이전
        # 전날 오전 11시 ~ 전날 오후 11시
        analysis_start = (now - timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)
        analysis_end = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
    elif now.hour < 23:  # 현재 시간이 오전 11시 이후, 오늘 오후 11시 이전
        # 전날 오후 11시 ~ 오늘 오전 11시
        analysis_start = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
        analysis_end = now.replace(hour=11, minute=0, second=0, microsecond=0)
    else:  # 현재 시간이 오후 11시 이후
        # 오늘 오전 11시 ~ 오늘 오후 11시
        analysis_start = now.replace(hour=11, minute=0, second=0, microsecond=0)
        analysis_end = now.replace(hour=23, minute=0, second=0, microsecond=0)



    # top_news를 정의하는 방법 두가지

    # 1. 클러스터링을 이용한 방법
    
    # target_date = datetime(2024, 12, 4)

    # # 날짜의 시작과 끝 정의
    # start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    # end_of_day = start_of_day + timedelta(days=1) - timedelta(seconds=1)

    # # 필터링: upload_date가 2024-11-21에 해당하는 데이터
    # all_news = YouTubeData.objects.filter(upload_date__gte=start_of_day, upload_date__lte=end_of_day)

    # titles = [news.title for news in all_news]
    # views = [news.views for news in all_news]
    # ids = [news._id for news in all_news]
    # texts = [news.desc for news in all_news]

    # top_news_ids = choose_10(titles,views,ids, texts)

    
    # # 해당 인스턴스의 id를 사용하여 새로운 QuerySet 생성
    # top_news = YouTubeData.objects.filter(_id__in=top_news_ids)


    # 2. 단순 조회순 정렬 (하루전체)
    target_date = datetime(2024, 12, 4)
    start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1) - timedelta(seconds=1)
    top_news = YouTubeData.objects.filter(upload_date__gte=start_of_day, upload_date__lte=end_of_day).order_by('-views')[:10]
    
    # 3. 오전오후
    top_news = YouTubeData.objects.filter(upload_date__gte=analysis_start, upload_date__lte=analysis_end).order_by('-views')[:10]

    # 제목 정리 및 찜 상태 확인
    for news in top_news:
        news.title = clean_title(news.title)

    for item in top_news:
        if request.user.is_authenticated:  # 로그인한 사용자인 경우에만 확인
            item.is_liked_by_user = item.like_set.filter(user=request.user).exists()
        else:
            item.is_liked_by_user = False

    # top_news 리스트의 각 항목에 대해 id 필드 추가
    for item in top_news:
        item.id = str(item._id)  # 직접 _id 값을 id로 할당
    
    context = {
        'section': 'chart',
        'top_news': top_news,
        'analysis_start': analysis_start,
        'analysis_end': analysis_end
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

# KST 타임존 처리 함수
def kst_to_aware(date_string):
    try:
        # 문자열을 datetime 객체로 파싱
        dt = parse(date_string)

        # 타임존이 이미 설정된 경우 그대로 반환
        if is_aware(dt):
            return dt

        # KST 문자열이 포함된 경우 UTC+9 적용
        if "KST" in date_string:
            dt = dt.replace(tzinfo=pytz.timezone("Asia/Seoul"))
        else:
            dt = make_aware(dt)  # 타임존이 없으면 UTC로 설정
        return dt

    except Exception as e:
        print(f"Failed to parse date: {date_string}, Error: {e}")
        return None


def save_all_historical_top10():
    all_videos = YouTubeData.objects.all().order_by('upload_date')  # 모든 비디오 정렬
    grouped_videos = defaultdict(list)

    for video in all_videos:
        try:
            # upload_date가 문자열이면 KST 타임존 적용
            if isinstance(video.upload_date, str):
                video.upload_date = kst_to_aware(video.upload_date)
                if not video.upload_date:  # 파싱 실패 시 건너뜀
                    continue
            elif not isinstance(video.upload_date, datetime):
                print(f"Invalid upload_date format for video ID {video._id}")
                continue  # 잘못된 형식이면 건너뜀

            # views를 정수형으로 변환
            if isinstance(video.views, str):
                video.views = int(video.views)

            date_key = video.upload_date.date()
            grouped_videos[date_key].append(video)

        except Exception as e:
            print(f"Error processing video ID {getattr(video, '_id', 'Unknown')}: {e}")

    # 각 날짜별로 상위 10개 저장
    for date_key, videos in grouped_videos.items():
        # 정렬 시 views를 기준으로 비교 (정수형)
        top_videos = sorted(videos, key=lambda x: x.views, reverse=True)[:10]
        for video in top_videos:
            WeeklyIssue.objects.update_or_create(
                _id=video._id,
                defaults={
                    'title': video.title,
                    'channel_name': video.channel_name,
                    'views': video.views,
                    'upload_date': video.upload_date,
                    'url': video.url,
                    'channel': video.channel,
                    'thumbnail': video.thumbnail,
                    'comments': video.comments if video.comments else [],
                    'transcript': video.transcript if video.transcript else []
                }
            )
    print("All historical top 10 videos saved successfully.")

def save_daily_top10():
    """
    어제 날짜 기준으로 조회수 상위 10개 데이터를 WeeklyIssue에 저장하는 함수
    """
    # 한국 시간 기준으로 오늘과 어제 날짜 계산
    today = localtime().date()
    yesterday = today - timedelta(days=1)

    # 어제 날짜의 상위 10개 영상 조회
    daily_videos = YouTubeData.objects.filter(
        upload_date__date=yesterday
    ).order_by('-views')[:10]

    for video in daily_videos:
        try:
            # WeeklyIssue에 저장 (중복 방지)
            WeeklyIssue.objects.update_or_create(
                _id=video._id,  # 기존 레코드 업데이트
                defaults={
                    'title': video.title,
                    'views': video.views,
                    'upload_date': video.upload_date,
                    'url': video.url,
                    'channel': video.channel,
                    'thumbnail': video.thumbnail,
                    'comments': video.comments if video.comments else [],
                    'transcript': video.transcript if video.transcript else []
                }
            )
        except Exception as e:
            print(f"Error while saving daily top 10 videos: {str(e)}")

def weekly_issues(request):
    """
    특정 날짜나 최근 7일간의 이슈 데이터를 보여주는 뷰 함수
    """
    # 한국 시간 기준으로 오늘 날짜와 7일 전 날짜 계산
    today = localtime().date()
    week_ago = today - timedelta(days=7)

    # GET 요청으로 특정 날짜 받기
    date_str = request.GET.get('date')
    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            target_date = None
    else:
        target_date = None

    # 특정 날짜가 있을 때
    if target_date:
        day_start = make_aware(datetime.combine(target_date, datetime.min.time()), timezone=pytz.timezone('Asia/Seoul'))
        day_end = make_aware(datetime.combine(target_date + timedelta(days=1), datetime.min.time()), timezone=pytz.timezone('Asia/Seoul'))

        daily_videos = WeeklyIssue.objects.filter(
            upload_date__gte=day_start,
            upload_date__lt=day_end
        ).order_by('-views')

        context = {
            'daily_videos': daily_videos,
            'target_date': target_date,
        }
        return render(request, 'analysis/weekly_issues.html', context)

    # 최근 7일 데이터 가져오기
    else:
        week_start = make_aware(datetime.combine(week_ago, datetime.min.time()), timezone=pytz.timezone('Asia/Seoul'))
        week_end = make_aware(datetime.combine(today + timedelta(days=1), datetime.min.time()), timezone=pytz.timezone('Asia/Seoul'))

        weekly_videos = WeeklyIssue.objects.filter(
            upload_date__gte=week_start,
            upload_date__lt=week_end
        ).order_by('-upload_date', '-views')

        grouped_issues = {}
        for video in weekly_videos:
            date_key = video.upload_date.astimezone(pytz.timezone('Asia/Seoul')).date()  # 날짜 변환만 적용
            if date_key not in grouped_issues:
                grouped_issues[date_key] = []
            grouped_issues[date_key].append(video)

        # 디버깅: grouped_issues 검증
        for date_key, videos in grouped_issues.items():
            print(f"Date: {date_key}, Count: {len(videos)}")  # 날짜별 데이터 개수 확인

        if 'all' in request.GET:
            sorted_issues = [(date_key, videos) for date_key, videos in grouped_issues.items()]
        else:
            sorted_issues = [(date_key, videos[:10]) for date_key, videos in grouped_issues.items()]

        sorted_issues = sorted(sorted_issues, key=lambda x: x[0], reverse=True)

        context = {
            'sorted_issues': sorted_issues,
        }
        return render(request, 'analysis/weekly_issues.html', context)

#상세분석
# @login_required
def detail(request):
    video_url = request.GET.get('url')
    video_id = request.GET.get('id')

    # 데이터베이스에서 값 가져오기
    video = YouTubeData.objects.filter(url=video_url).first()
    video_views = video.views if video else None  # 조회수
    video_likes = video.likes if video else None  # 좋아요 수
    video_comments = video.comments if video else None  # 댓글 수
    video_title = video.title if video else None  # 동영상 제목
    # 컨텍스트 구성
    context = {
        'video': video,
        'video_id': video_id,
        'video_url': video_url,
        'video_title': video_title,
        'video_views': video_views,
        'video_likes': video_likes,
        'video_comments': video_comments,
    }

    return render(request, 'analysis/detail.html', context)



import matplotlib.pyplot as plt
from io import BytesIO
import base64
from wordcloud import WordCloud
from transformers import pipeline
from django.shortcuts import render
from collections import Counter
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

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.conf import settings


def password_reset_request(request):
    if request.method == "POST":
        form = PasswordResetRequestForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            try:
                user = User.objects.get(email=email)
                token = default_token_generator.make_token(user)
                uid = urlsafe_base64_encode(force_bytes(user.pk))
                reset_url = f"{request.scheme}://{request.get_host()}/password-reset/{uid}/{token}/"

                # 이메일 전송
                send_mail(
                    '비밀번호 재설정 요청',
                    f'비밀번호를 재설정하려면 다음 링크를 클릭하세요: {reset_url}',
                    settings.DEFAULT_FROM_EMAIL,
                    [email],
                )
                return render(request, 'registration/password_reset_done.html')
            except User.DoesNotExist:
                form.add_error('email', '해당 이메일이 등록되어 있지 않습니다.')
    else:
        form = PasswordResetRequestForm()

    return render(request, 'registration/password_reset_request.html', {'form': form})

def password_reset_confirm(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        if request.method == "POST":
            form = SetNewPasswordForm(request.POST)
            if form.is_valid():
                user.set_password(form.cleaned_data['new_password'])
                user.save()
                return render(request, 'registration/password_reset_complete.html')
        else:
            form = SetNewPasswordForm()
        return render(request, 'registration/password_reset_confirm.html', {'form': form})
    else:
        return render(request, 'registration/password_reset_invalid.html')