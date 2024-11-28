'''
pip install konlpy networkx matplotlib pandas
'''
from collections import defaultdict
from datetime import timedelta

from django.db.models.functions import TruncDate
from django.http import JsonResponse
from django.utils import timezone
from django.shortcuts import render, get_object_or_404
from django.utils.timezone import localtime
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import date

from .forms import UserRegistrationForm
from .models import YouTubeData
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

# @login_required
def chart(request): # 차트 뷰
    # 현재 시간 가져오기
    now = timezone.now()
    # 오늘 오후 6시 10분 (KST) 계산
    today_615pm_kst = timezone.localtime().replace(hour=18, minute=15, second=0, microsecond=0)
    # 어제 오후 6시 10분 (KST) 계산
    yesterday_610pm_kst = today_615pm_kst - timedelta(days=1)

    # KST를 UTC로 변환 (UTC = KST-9)
    yesterday_610pm_utc = yesterday_610pm_kst - timedelta(hours=9)
    today_610pm_utc = today_615pm_kst - timedelta(hours=9)

    # 쿼리 실행: 어제 오후 6시 15분 ~ 오늘 오후 6시 15분 사이에 업로드된 동영상 가져오기
    # 조회수 기준 내림차순 정렬, 상위 10개 데이터 가져오기
    top_news = YouTubeData.objects.filter(
        upload_date__gte=yesterday_610pm_utc, # 어제 오후 6시 15분 이후 데이터
        upload_date__lte=today_610pm_utc  # 오늘 오후 6시 15분 이전 데이터
    ).order_by('-views')[:10]

    # 제목 필터링 적용
    for news in top_news:
        news.title = clean_title(news.title)

    # 템플릿에 데이터 전달
    return render(request, 'analysis/chart.html', {'top_news': top_news})

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

def analyze_sentiment(comments):
    sentiment_results = []
    for comment in comments:
        result = sentiment_analysis_pipeline(comment)[0]
        sentiment_results.append({
            'comment': comment,
            'sentiment': result['label'],  # 'POSITIVE' 또는 'NEGATIVE'
            'confidence': result['score']
        })
    return sentiment_results

def generate_wordcloud(comments, analyzed_comments):
    text = " ".join(comments)

    if not text.strip():
        raise ValueError("No words available to generate WordCloud.")

    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 시스템에 설치된 경로 예시

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        for sentiment in analyzed_comments:
            if sentiment['comment'] and word in sentiment['comment']:
                if sentiment['sentiment'] == 'POSITIVE':
                    return 'rgb(0, 0, 255)'  # 파란색 (긍정)
                else:
                    return 'rgb(255, 0, 0)'  # 빨간색 (부정)
        return 'rgb(0, 0, 0)'  # 기본 색상 (검정)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        font_path=font_path,  # 한글 폰트 경로
        color_func=color_func
    ).generate(text)

    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return img_base64

def generate_pie_chart(analyzed_comments):
    positive_count = sum(1 for comment in analyzed_comments if comment['sentiment'] == 'POSITIVE')
    negative_count = sum(1 for comment in analyzed_comments if comment['sentiment'] == 'NEGATIVE')

    labels = ['Positive', 'Negative']
    sizes = [positive_count, negative_count]
    colors = ['#66b3ff', '#ff6666']
    explode = (0.1, 0)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return img_base64

def generate_rank_table_by_morpheme(analyzed_comments):
    okt = Okt()

    # 댓글에서 형태소 분석을 통해 명사만 추출
    words = []
    for comment in analyzed_comments:
        morphemes = okt.nouns(comment['comment'])
        words.extend(morphemes)

    # 단어 빈도수 계산
    word_counts = Counter(words)

    # 빈도수가 높은 순으로 정렬된 상위 10개의 단어 추출
    sorted_words = word_counts.most_common(10)

    # 표 형식으로 반환
    return sorted_words


def emotion(request):
    video_url = request.GET.get('url')
    video_title = request.GET.get('title')
    video_id = request.GET.get('id')

    video = YouTubeData.objects.filter(url=video_url).first()

    video_comments = []
    if video and isinstance(video.comments, list):
        for comment in video.comments:
            if isinstance(comment, dict):
                video_comments.append(comment.get('comment'))

    if not video_comments:
        video_comments = ["No comments available"]

    analyzed_comments = analyze_sentiment(video_comments)

    try:
        wordcloud_image = generate_wordcloud(video_comments, analyzed_comments)
    except ValueError as e:
        wordcloud_image = None

    pie_chart_image = generate_pie_chart(analyzed_comments)

    # 형태소 분석을 통한 빈도수 차트 생성 (표 형식)
    rank_table_by_morpheme = generate_rank_table_by_morpheme(analyzed_comments)

    video_title = video.title if video else video_title

    context = {
        'section': 'emotion',
        'video_title': video_title,
        'video_comments': video_comments,
        'analyzed_comments': analyzed_comments,
        'video_url': video_url,
        'video_id': video_id,
        'wordcloud_image': wordcloud_image,
        'pie_chart_image': pie_chart_image,
        'rank_table_by_morpheme': rank_table_by_morpheme,  # 형태소 분석을 통한 표 추가
    }

    return render(request, 'analysis/emotion.html', context)


def analyze_related_words(text):
    okt = Okt()
    
    # 명사 추출
    nouns = okt.nouns(text)
    
    # 2글자 이상의 명사만 선택
    words = [word for word in nouns if len(word) > 1]
    
    # 단어 쌍 생성
    word_pairs = []
    for i in range(len(words)-1):
        for j in range(i+1, min(i+5, len(words))):
            word_pairs.append(tuple(sorted([words[i], words[j]])))
    
    # 단어 쌍 빈도수 계산
    pair_counts = Counter(word_pairs)
    
    # 네트워크 그래프 생성
    G = nx.Graph()
    
    # 상위 30개의 연관 관계만 사용
    for (word1, word2), count in pair_counts.most_common(30):
        G.add_edge(word1, word2, weight=count)
    
    return G, pair_counts.most_common(30)

def generate_network_graph(G):
    # matplotlib 백엔드 설정
    matplotlib.use('Agg')
    
    # 한글 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 환경
    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rc('font', family=font_name)
    
    # 그래프 크기 조절 (더 작게 설정)
    plt.figure(figsize=(8, 6))  # 원래 (12, 8)에서 변경
    
    # 노드 크기 설정 (더 작게 조절)
    node_size = [G.degree(node) * 200 for node in G.nodes()]  # 300에서 200으로 변경
    
    # 엣지 굵기 설정 (더 얇게 조절)
    edge_width = [G[u][v]['weight'] * 0.3 for u, v in G.edges()]  # 0.5에서 0.3으로 변경
    
    # 그래프 레이아웃 설정 (노드 간격 조절)
    pos = nx.spring_layout(G, k=0.8, iterations=50)  # k값을 1에서 0.8로 변경
    
    # 그래프 그리기
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.4)
    
    # 한글 레이블 설정 (폰트 크기 조절)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_family=font_name, font_size=8)  # 10에서 8로 변경
    
    plt.title('연관어 네트워크', fontdict={'family': font_name, 'size': 12}, pad=20)
    plt.axis('off')
    
    # 그래프를 이미지로 변환
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=200, facecolor='white')  # dpi 300에서 200으로 변경
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # base64로 인코딩
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    plt.close()
    
    return graphic

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


def register(request):
    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        if user_form.is_valid():
            # Create a new user object but avoid saving it yet
            new_user = user_form.save(commit=False)
            # Set the chosen password
            new_user.set_password(user_form.cleaned_data['password'])
            # Save the User object
            new_user.save()
            return render(request, 'registration/register_done.html', {'new_user': new_user})
    else:
        user_form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'user_form': user_form})
