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
from .models import YouTubeData, Like, WeeklyIssue, Chart
from urllib.parse import urlparse, parse_qs

from pytz import timezone
import pytz

import matplotlib
matplotlib.use('Agg')
from django.utils.timezone import localtime, utc


from .analysis.relate_analysis import analyze_related_words, generate_network_graph
from .analysis.emotion_analysis import (
    generate_wordcloud,
    generate_pie_chart,
    analyze_morphemes,
    analyze_sentiment
)
from django.contrib.auth.models import User

from sklearn.metrics.pairwise import cosine_similarity
from .analysis.text_processing import clean_title, process_text, get_top10_chart_based
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from django.utils.timezone import localtime
from datetime import timedelta

from .analysis.visualization import visualize_performance_metrics

def save_top_videos(start_time, end_time, model):
    """
    특정 시간대의 상위 10개 동영상을 지정된 모델에 저장
    :param start_time: 데이터 필터링 시작 시간
    :param end_time: 데이터 필터링 종료 시간
    :param model: 데이터를 저장할 Django 모델 (Chart, WeeklyIssue 등)
    """
    try:
        # 데이터베이스에서 시간 범위에 해당하는 데이터 가져오기
        all_data = YouTubeData.objects.filter(
            upload_date__gte=start_time,
            upload_date__lt=end_time
        ).order_by('-views')

        if not all_data.exists():
            print(f"해당 시간 범위에 데이터가 없습니다. {start_time} ~ {end_time}")
            return

        # 상위 10개 데이터 선정
        top_videos = get_top10_chart_based(all_data)

        # 상위 10개 데이터를 지정된 모델에 저장
        for rank, video in enumerate(top_videos, start=1):
            try:
                video_id = ObjectId(video._id) if isinstance(video._id, str) else video._id
                model.objects.update_or_create(
                    _id=video_id,  # MongoDB ObjectId
                    defaults={
                        "chart_date": localtime(),
                        "rank": rank,
                        "channel_name": video.channel_name,
                        "title": video.title,
                        "views": video.views,
                        "upload_date": video.upload_date,
                        "url": video.url,
                        "channel": video.channel,
                        "desc": video.desc,
                        "likes": video.likes,
                        "thumbnail": video.thumbnail,
                        "comments": video.comments,
                        "transcript": video.transcript,
                    }
                )
            except Exception as e:
                print(f"Error saving video {video._id}: {e}")

        print(f"데이터 저장 완료: {start_time} ~ {end_time}")
    except Exception as e:
        print(f"Error in save_top_videos: {e}")

def save_top10_to_chart():
    """
    상위 10개의 동영상 데이터를 Chart 컬렉션에 저장.
    """
    try:
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

        # 시간대를 UTC로 변환
        analysis_start = analysis_start.astimezone(utc)
        analysis_end = analysis_end.astimezone(utc)

        print(f"Analysis start(UTC): {analysis_start}, Analysis end(UTC): {analysis_end}")

        # 데이터 저장 로직
        save_top_videos(analysis_start, analysis_end, Chart)

    except Exception as e:
        print(f"Error in save_top10_to_chart: {e}")

import logging

logger = logging.getLogger('chart_cleanup')

def delete_expired_charts():
    """
    Chart 데이터에서 24시간이 지난 항목만 삭제
    """
    try:
        # 현재 시간 기준으로 24시간 전 시간 계산
        now = localtime()
        expiration_time = now - timedelta(hours=24)

        # 24시간 이전의 데이터를 필터링하여 삭제
        expired_charts = Chart.objects.filter(chart_date__lt=expiration_time)
        deleted_count, _ = expired_charts.delete()

        logger.info(f"{deleted_count}개의 24시간 지난 Chart 데이터가 삭제되었습니다.")
    except Exception as e:
        logger.error(f"delete_expired_charts 실행 중 오류 발생: {e}")

def chart(request):
    """
    Chart 데이터를 템플릿으로 전달
    """
    try:
        now = localtime()

        # 기준 시간 설정
        if now.hour < 11:
            analysis_start = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
            analysis_end = now.replace(hour=11, minute=0, second=0, microsecond=0)
        elif now.hour < 23:
            analysis_start = now.replace(hour=11, minute=0, second=0, microsecond=0)
            analysis_end = now.replace(hour=23, minute=0, second=0, microsecond=0)
        else:
            analysis_start = now.replace(hour=23, minute=0, second=0, microsecond=0)
            analysis_end = (now + timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)

        analysis_start_utc = analysis_start.astimezone(utc)
        analysis_end_utc = analysis_end.astimezone(utc)

        # MongoDB에서 데이터 필터링
        chart_data = Chart.objects.filter(
            chart_date__gte=analysis_start_utc,
            chart_date__lt=analysis_end_utc
        )
        chart_data = sorted(chart_data, key=lambda x: x.rank)

        # QuerySet 개수를 확인하려면 .count()를 사용
        print("DEBUG: QuerySet count before sorting:", Chart.objects.filter(
            chart_date__gte=analysis_start_utc,
            chart_date__lt=analysis_end_utc
        ).count())

        # 리스트의 길이를 확인하려면 len()을 사용
        print("DEBUG: chart_data count after sorting:", len(chart_data))

        processed_chart_data = []
        for chart in chart_data:
            try:
                processed_chart_data.append({
                    "rank": chart.rank,
                    "title": chart.title,
                    "cleaned_title": clean_title(chart.title),
                    "views": chart.views,
                    "channel_name": chart.channel_name,
                    "url": chart.url,
                    "upload_date": chart.upload_date,
                    "thumbnail": chart.thumbnail,
                    "id": str(chart._id)  # _id를 id로 매핑
                })
            except Exception as e:
                print(f"Error processing chart data: {e}")

        # 템플릿에 전달할 데이터 구성
        context = {
            "top_news": processed_chart_data,
            "analysis_start": analysis_start,
            "analysis_end": analysis_end
        }

        print("DEBUG: context", context)  # 전달 데이터 확인
        return render(request, 'analysis/chart.html', context)
    except Exception as e:
        import traceback
        print(f"Error in chart function: {e}")
        print(traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt  # CSRF 검사 비활성화(POST 요청 허용)
def video_details(request):
    """
    Chart 컬렉션에서 특정 동영상의 세부 정보를 반환하는 API
    """
    if request.method == 'POST':
        try:
            # 요청 본문에서 JSON 데이터 로드
            data = json.loads(request.body)
            video_url = data.get('url')  # JSON 데이터에서 URL 추출

            # MongoDB Chart 컬렉션에서 URL로 데이터 검색
            video = Chart.objects.filter(url=video_url).first()

            # 검색된 데이터가 없을 경우 에러 응답
            if not video:
                return JsonResponse({"error": "Video not found in the chart."}, status=404)

            # 검색된 데이터가 있을 경우 JSON 응답 생성
            response_data = {
                "video_url": video.url,
                "video_title": video.title,
                "video_id": parse_qs(urlparse(video.url).query).get('v', [None])[0] or video.url.split('/')[-1],  # 유연한 ID 추출
                "views": video.views,
                "likes": video.likes,
                "thumbnail": video.thumbnail,
                "upload_date": video.upload_date.isoformat() if video.upload_date else None,
                "channel_name": video.channel_name,
                "description": video.desc,
                "comments": video.comments if isinstance(video.comments, list) else [],
                "transcript": video.transcript if isinstance(video.transcript, list) else [],
            }

            return JsonResponse(response_data, safe=False, status=200)

        except Exception as e:
            print(f"Error in video_details function: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    # 요청이 POST 방식이 아닐 경우 에러 응답
    return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)

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

def save_daily_top10():
    """
    어제 날짜 데이터를 기반으로 상위 10개 비디오를 WeeklyIssue 컬렉션에 저장
    """
    try:
        seoul_tz = timezone('Asia/Seoul')  # 한국 시간대
        today = datetime.now(seoul_tz).date()
        yesterday = today - timedelta(days=1)

        start_time = datetime.combine(yesterday, datetime.min.time()).replace(tzinfo=seoul_tz).astimezone(pytz.UTC)
        end_time = datetime.combine(yesterday, datetime.max.time()).replace(tzinfo=seoul_tz).astimezone(pytz.UTC)

        save_top_videos(start_time, end_time, WeeklyIssue)
    except Exception as e:
        print(f"Error in save_daily_top10: {e}")

def weekly_issues(request):
    date_str = request.GET.get('date')
    seoul_tz = pytz.timezone('Asia/Seoul')
    today = datetime.now(seoul_tz).date()
    yesterday = today - timedelta(days=1)

    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            target_date = yesterday
    else:
        target_date = yesterday

    if target_date > yesterday:
        target_date = yesterday

    start_date = target_date - timedelta(days=5)
    end_date = target_date + timedelta(days=1)

    start_date_utc = datetime.combine(start_date, datetime.min.time()).astimezone(pytz.UTC)
    end_date_utc = datetime.combine(end_date, datetime.min.time()).astimezone(pytz.UTC)

    try:
        issues = WeeklyIssue.objects.filter(
            upload_date__gte=start_date_utc,
            upload_date__lt=end_date_utc
        )
        issues = sorted(issues, key=lambda x: x.upload_date, reverse=True)

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

        sorted_issues = sorted(grouped_issues.items(), key=lambda x: x[0], reverse=True)[:6]

        context = {
            'sorted_issues': sorted_issues,
            'target_date': target_date,
            'yesterday': yesterday,
        }
        return render(request, 'analysis/weekly_issues.html', context)

    except Exception as e:
        print(f"Error in weekly_issues function: {e}")
        return JsonResponse({"error": str(e)}, status=500)

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

    ####피드백 폼 처리
    if request.method == 'POST':
        feedback_form = FeedbackForm(request.POST)
        if feedback_form.is_valid():
            feedback = feedback_form.save(commit=False)
            feedback.user = request.user
            feedback.save()
            return redirect('/account/feedback_done/')
    else:
        feedback_form = FeedbackForm()

    context = {
        'form': feedback_form,
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
            # 제목과 설명 불용어 처리
            cleaned_title = clean_title(video.title)
            video_desc = f"{cleaned_title} {video.desc if video.desc else ''}"
            
            # transcript 데이터 처리
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
            
            # 성능 평가를 제외한 기본 분석 수행
            graph, top_pairs, important_keywords, _ = analyze_related_words(
                video_desc, 
                video.transcript,
                clean_title_func=clean_title
            )
            
            network_graph = generate_network_graph(graph)
            
            # 키워드별 관련 뉴스 분류
            categorized_news = {}
            if important_keywords:
                for keyword in important_keywords[:10]:  # 상위 10개 키워드만 처리
                    related_news = YouTubeData.objects.filter(
                        Q(title__icontains=keyword) | 
                        Q(desc__icontains=keyword)
                    ).exclude(url=video_url)[:6]  # 뉴스 개수 제한
                    
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
                'transcript_segments': transcript_segments
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
    subject = "비밀번호 초기화 내"
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


from django.shortcuts import render, redirect
from .models import Feedbacks 
from .forms import FeedbackForm
def feedback(request):
    if request.method == 'POST':
        feedback_form = FeedbackForm(request.POST)
        if feedback_form.is_valid():
            feedback = feedback_form.save(commit=False)
            feedback.user = request.user
            feedback.save()
            return redirect('feedback_done')
    else:
        feedback_form = FeedbackForm()
    return render(request, 'feedback/feedback.html', {'feedback_form': feedback_form})



def feedback_done(request):
    return render(request, 'feedback/feedback_done.html')
