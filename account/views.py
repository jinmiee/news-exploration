'''
pip install konlpy networkx matplotlib pandas
'''
from collections import defaultdict
from datetime import timedelta

from django.db.models.functions import TruncDate
from django.http import JsonResponse
from django.utils import timezone
from django.shortcuts import render, get_object_or_404, redirect
from django.utils.timezone import localtime
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import date
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseBadRequest
from bson import ObjectId


from .forms import UserRegistrationForm
from .models import YouTubeData, like
from urllib.parse import urlparse, parse_qs
import re

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


def clean_title(title):
    """
    뉴스 제목에서 /방송사이름만 제거하고 나머지 텍스트는 유지하는 함수
    """
    # 대괄호 안의 내용 제거 (예: [이슈])
    title = re.sub(r'\[.*?\]', '', title)

    # 슬래시(/)로 시작하는 날짜 제거 (예: /2024년 11월 26일)
    title = re.sub(r'/\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일', '', title)

    # 괄호와 그 안의 내용 제거 (예: (2024.11.27))
    title = re.sub(r'\(.*?\)', '', title)

    # 특정 단어 제거 (예: TV, News, 8뉴스, 오대영 라이브)
    title = re.sub(r'\b(TV|News|8뉴스|오대영 라이브)\b', '', title, flags=re.IGNORECASE)

    # 특정 단어 제거 (예: TV)
    title = re.sub(r'\bTV\b', '', title, flags=re.IGNORECASE)

    # /방송사 이름 제거 (예: /YTN, /JTBC, /KBS)
    title = re.sub(r'/\s*(YTN|JTBC|KBS|MBC|연합뉴스|SBS)\s*', '', title, flags=re.IGNORECASE)

    # 해시태그 제거 (예: #뉴스다)
    title = re.sub(r'#\S+', '', title)

    # 다중 공백 제거 (괄호 제거 후 남을 수 있는 공백 처리)
    title = re.sub(r'\s+', ' ', title)

    # 앞뒤 공백 제거
    return title.strip()


@login_required
def chart(request):
    """
    실시간 뉴스 차트 뷰: 전날 23시 ~ 오늘 11시 기준으로 데이터 조회
    """
    now = localtime()

    # 현재 시간이 오전 11시 이전인지 확인
    if now.hour < 11:
        # 현재 시간이 오전 11시 이전이면, 어제 오후 11시부터 오늘 오전 11시까지
        analysis_start = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
        analysis_end = now.replace(hour=11, minute=0, second=0, microsecond=0)
    else:
        # 현재 시간이 오전 11시 이후면, 오늘 오전 11시부터 내일 오전 11시까지
        analysis_start = now.replace(hour=11, minute=0, second=0, microsecond=0)
        analysis_end = now.replace(hour=23, minute=0, second=0, microsecond=0) + timedelta(hours=12)

    # 데이터 가져오기
    top_news = YouTubeData.objects.filter(
        upload_date__gte=analysis_start,
        upload_date__lt=analysis_end
    ).order_by('-views')[:10]

    # 제목 정리 및 찜 상태 확인
    for news in top_news:
        news.title = clean_title(news.title)

    for item in top_news:
        if request.user.is_authenticated:  # 로그인한 사용자인 경우에만 확인
            item.is_liked_by_user = item.like_set.filter(user=request.user).exists()
        else:
            item.is_liked_by_user = False

    # 컨텍스트에 데이터 전달
    context = {
        'top_news': top_news,
        'analysis_start': analysis_start,
        'analysis_end': analysis_end,
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



def weekly_issues(request):
    # 현재 날짜 가져오기
    today = localtime().date()
    # 오늘로부터 7일 전 날짜 계산
    week_ago = today - timedelta(days=7)

    # 데이터베이스에서 7일 동안 업로드된 동영상을 조회
    # 조건: 업로드 날짜가 7일 전 이후(>=)이고 오늘 이전(<=)
    # 정렬: 업로드 날짜 역순(-upload_date) 및 조회수 높은 순(-views)
    weekly_videos = YouTubeData.objects.filter(
        upload_date__gte=week_ago,  # 업로드 날짜가 7일 전 이후
        upload_date__lte=today      # 업로드 날짜가 오늘 이전
    ).order_by('-upload_date', '-views')  # 최신순 및 조회수 높은 순으로 정렬

    # 데이터를 날짜별로 그룹화
    grouped_issues = defaultdict(list)  # 기본값이 빈 리스트인 딕셔너리 생성
    for video in weekly_videos:  # weekly_videos 쿼리셋의 각 동영상에 대해 반복
        date_key = video.upload_date.date()  # 업로드 날짜(연-월-일) 추출
        # 날짜별로 최대 10개 동영상만 추가
        if len(grouped_issues[date_key]) < 10:
            grouped_issues[date_key].append(video)  # 날짜 키에 해당하는 리스트에 동영상 추가

    # grouped_issues 딕셔너리를 날짜 순서로 정렬
    # 정렬 기준: 날짜 (x[0]), 내림차순(reverse=True)
    sorted_issues = sorted(grouped_issues.items(), key=lambda x: x[0], reverse=True)

    # 정렬된 데이터를 템플릿에 전달
    # sorted_issues는 (날짜, [동영상 리스트]) 형태의 리스트
    return render(request, 'analysis/weekly_issues.html', {'sorted_issues': sorted_issues})



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

def relate(request):
    video_url = request.GET.get('url')
    video_id = request.GET.get('id')
    
    # 데이터베이스에서 비디오 정보 가져오기
    video = YouTubeData.objects.filter(url=video_url).first()
    
    if video and video.desc:
        # 연관어 분석 수행
        graph, top_pairs = analyze_related_words(video.desc)
        
        # 네트워크 그래프 생성
        network_graph = generate_network_graph(graph)
        
        context = {
            'section': 'relate',
            'video': video,
            'video_id': video_id,
            'network_graph': network_graph,
            'top_pairs': top_pairs,
            'video_title': video.title
        }
    else:
        context = {
            'section': 'relate',
            'error_message': '비디오 설명이 없거나 비디오를 찾을 수 없습니다.'
        }
    
    return render(request, 'analysis/relate.html', context)


# @login_required
def mypage(request):
    return render(request, 'analysis/mypage/mypage.html', {'section': 'mypage'}) 


# @login_required
def like_video(request, id):
    # 문자열을 ObjectId로 변환
    print(id)
    print(type(id))
    try:
        video = YouTubeData.objects.get(id=id)
        if Like.objects.filter(user=request.user, video=video).exists():
            Like.objects.filter(user=request.user, video=video).delete()
            is_liked = False
        else:
            Like.objects.create(user=request.user, video=video)
            is_liked = True
        return JsonResponse({'status': 'success', 'is_liked': is_liked})
    except YouTubeData.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '동영상을 찾을 수 없습니다.'})

# @login_required
def my_liked_videos(request):
    liked_videos = YouTubeData.objects.filter(like__user=request.user)  
    
    context = {
        'liked_videos': liked_videos,
        'section': 'mypage'
    } 
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
