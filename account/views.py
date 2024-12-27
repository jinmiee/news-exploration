'''
pip install konlpy networkx matplotlib pandas
'''
from collections import defaultdict
from datetime import timedelta, datetime

from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.views import PasswordChangeView
from django.http import JsonResponse
from django.urls import reverse_lazy
from django.utils import timezone

from django.utils.timezone import localtime, make_aware, is_aware
from django.views.decorators.csrf import csrf_exempt
import json
from django.contrib.auth.decorators import login_required
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer

from .forms import UserRegistrationForm, CustomPasswordChangeForm
from .models import YouTubeData, Like, WeeklyIssue, Chart, WeeklyIssueDuplicateVideo, ChartDuplicateVideo
from urllib.parse import urlparse, parse_qs

from pytz import timezone
import pytz

import matplotlib
matplotlib.use('Agg')
from django.utils.timezone import localtime, utc


from .analysis.relate_analysis import analyze_related_words

from django.contrib.auth.models import User

from sklearn.metrics.pairwise import cosine_similarity
from .analysis.text_processing import clean_title, process_text, get_top10_chart_based
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from django.utils.timezone import localtime
from datetime import timedelta
from .models import Like, YouTubeData, WeeklyIssue, Chart, WeeklyIssueDuplicateVideo, ChartDuplicateVideo
from .analysis.visualization import generate_network_graph
from .tasks.processing_tasks import (
    save_top_videos,
    save_top10_to_chart,
    delete_expired_charts,
    save_daily_top10,
    extract_duplicates_for_weekly_issues,
    extract_duplicates_for_chart,
)

def chart(request):
    """
    Chart 데이터를 템플릿으로 전달
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

        analysis_start_utc = analysis_start.astimezone(utc)
        analysis_end_utc = analysis_end.astimezone(utc)

        # MongoDB에서 데이터 필터링
        chart_data = Chart.objects.filter(
            upload_date__gte=analysis_start_utc,
            upload_date__lt=analysis_end_utc
        )
        chart_data = sorted(chart_data, key=lambda x: x.rank)

        # QuerySet 개수를 확인하려면 .count()를 사용
        print("DEBUG: QuerySet count before sorting:", Chart.objects.filter(
            upload_date__gte=analysis_start_utc,
            upload_date__lt=analysis_end_utc
        ).count())

        # 리스트의 길이를 확인하려면 len()을 사용
        print("DEBUG: chart_data count after sorting:", len(chart_data))
        print(chart_data)
        processed_chart_data = []
        for chart in chart_data:
            try:
                processed_chart_data.append({
                    "rank": chart.rank,
                    "title": chart.title,
                    "cleaned_title": clean_title(chart.title),
                    "views": chart.views,
                    "likes": chart.likes,
                    "channel_name": chart.channel_name,
                    "url": chart.url,
                    "upload_date": chart.upload_date,
                    "thumbnail": chart.thumbnail,
                    "id": str(chart._id),  # _id를 id로 매핑
                    
                })
                if request.user.is_authenticated:
                    processed_chart_data[-1]['is_liked_by_user'] = Like.objects.filter(user=request.user, youtube_data=YouTubeData.objects.get(_id = chart._id)).exists()
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

from fuzzywuzzy import fuzz
from datetime import timedelta

def get_related_duplicate_videos(request):
    try:
        # 요청에서 URL을 가져옵니다.
        video_url = request.GET.get('url')
        if not video_url:
            return JsonResponse({"error": "URL 매개변수가 제공되지 않았습니다."}, status=400)

        # WeeklyIssue 또는 Chart에서 해당 URL과 일치하는 동영상 가져오기
        video = WeeklyIssue.objects.filter(url=video_url).first() or Chart.objects.filter(url=video_url).first()
        if not video:
            return JsonResponse({"error": "해당 URL에 대한 기사를 찾을 수 없습니다."}, status=404)

        # 중복 동영상 모델 선택
        if WeeklyIssue.objects.filter(url=video_url).exists():
            duplicates_model = WeeklyIssueDuplicateVideo
        elif Chart.objects.filter(url=video_url).exists():
            duplicates_model = ChartDuplicateVideo
        else:
            return JsonResponse({"error": "데이터 유형을 확인할 수 없습니다."}, status=400)

        # 중복 검색 (유사도 + 업로드 날짜 범위 기준)
        duplicates = []
        for candidate in duplicates_model.objects.all():
            similarity = fuzz.ratio(video.title, candidate.title)
            if similarity > 70 : # 유사도가 70% 이상이고 업로드 날짜가 하루 이내인 경우
                duplicates.append(candidate)

        # 디버깅 로그 출력
        print(f"DEBUG: 중복 동영상 개수: {len(duplicates)}")
        for duplicate in duplicates:
            print(f"중복 동영상: {duplicate.title}, URL: {duplicate.url}, 유사도: {similarity}")

        # JSON 응답 생성
        duplicate_list = [
            {
                "title": duplicate.title,
                "url": duplicate.url,
                "views": duplicate.views,
                "upload_date": duplicate.upload_date.isoformat(),
                "channel_name": duplicate.channel_name,
                "thumbnail": duplicate.thumbnail,
            }
            for duplicate in duplicates
        ]

        return JsonResponse({"duplicates": duplicate_list}, safe=False, status=200)

    except Exception as e:
        print(f"중복 동영상 검색 오류: {e}")
        return JsonResponse({"error": "중복 동영상 검색 중 오류가 발생했습니다."}, status=500)

#상세분석
# @login_required
from bson import ObjectId
from django.shortcuts import get_object_or_404
from django.http import JsonResponse

def detail(request, video_id=None):
    """
    video_id 또는 URL을 기반으로 비디오 데이터를 가져옵니다.
    """
    try:
        video = None
        object_id = None

        # URL 또는 ID로 접근 구분
        if video_id:
            try:
                # MongoDB ObjectId로 변환 및 데이터 조회
                object_id = ObjectId(video_id)
                video = get_object_or_404(YouTubeData, _id=object_id)
            except Exception as e:
                return JsonResponse({"error": f"Invalid video ID: {e}"}, status=400)
        else:
            # GET 파라미터에서 url 사용
            video_url = request.GET.get('url')
            if not video_url:
                return JsonResponse({"error": "URL 또는 video_id가 제공되지 않았습니다."}, status=400)

            video = YouTubeData.objects.filter(url=video_url).first()
            if not video:
                return JsonResponse({"error": "해당 URL에 대한 데이터를 찾을 수 없습니다."}, status=404)

        # 공통 처리 (video가 None인 경우를 방지)
        if video:
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

            # 연관어 분석 수행
            graph, top_pairs, important_keywords, _ = analyze_related_words(
                video_desc,
                video.transcript,
                clean_title_func=clean_title
            )

            network_graph = generate_network_graph(graph)

            # 키워드별 관련 뉴스 분류
            categorized_news = {}
            if important_keywords:
                for keyword in important_keywords[:5]:  # 상위 5개 키워드만 처리
                    related_news = YouTubeData.objects.filter(
                        Q(title__icontains=keyword) |
                        Q(desc__icontains=keyword)
                    ).exclude(_id=object_id)[:5]  # 현재 비디오 제외

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

            # 댓글 처리
            video_comments = []
            if video.comments:
                for comment in video.comments[:100]:  # 최대 100개만 처리
                    if isinstance(comment, dict):
                        comment_text = comment.get('comment', '')
                    else:
                        comment_text = str(comment)
                    if comment_text.strip():
                        video_comments.append(comment_text)

            # 댓글 분석 결과
            if video_comments:
                wordcloud_base64, pie_chart_base64 = save_visualizations_with_tfidf(video_comments)
                bubble_chart_base64 = save_bubble_chart_with_tfidf(video_comments)
                sentiment_html = generate_tfidf_sentiment_visualizations(video_comments)
            else:
                wordcloud_base64 = pie_chart_base64 = bubble_chart_base64 = sentiment_html = None

            # 템플릿에 전달할 데이터 구성
            context = {
                'video_url': video.url,
                'video_id': str(video._id),
                'video_comments': video.comments,
                'video_views': video.views,
                'video_likes': video.likes,
                'network_graph': network_graph,
                'categorized_news': categorized_news,
                'important_keywords': important_keywords,
                'transcript_segments': transcript_segments,
                'wordcloud_image': wordcloud_base64,
                'pie_chart_image': pie_chart_base64,
                'bubble_chart_image': bubble_chart_base64,  # 버블차트 추가
                'sentiment_table': sentiment_html  # TF-IDF 분석 결과 테이블 전달
            }

            return render(request, 'analysis/detail.html', context)

    except Exception as e:
        print(f"Error in detail function: {e}")
        return JsonResponse({"error": str(e)}, status=500)

from django.shortcuts import render
from django.http import HttpResponse
from .models import YouTubeData
from .analysis.emotion_analysis import save_visualizations_with_tfidf  # 워드클��우드 및 파이차트 생성 함수
from .analysis.emotion_analysis import generate_tfidf_sentiment_visualizations  # 감정 분석 결과 HTML 생성 함수
from .analysis.emotion_analysis import  save_bubble_chart_with_tfidf # 버블
def emotion(request):
    video_url = request.GET.get('url')
    print(f"요청된 비디오 URL: {video_url}")

    # 해당 비디오 URL에 대한 YouTube 데이터 조회
    video = YouTubeData.objects.filter(url=video_url).first()

    if video and video.comments:
        try:
            video_comments = []
            for comment in video.comments[:100]:
                if isinstance(comment, dict):
                    comment_text = comment.get('comment', '')
                else:  # dict가 아니면 그냥 문자열로 취급
                    comment_text = str(comment)

                if comment_text.strip():  # 빈 댓글 제외
                    video_comments.append(comment_text)

            if not video_comments:
                raise ValueError("유효한 댓글이 없습니다.")

            # TF-IDF 분석 후 시각화 (워드클라우드 및 파이차트)
            wordcloud_base64, pie_chart_base64 = save_visualizations_with_tfidf(video_comments)

            # TF-IDF 기반 감정 분석 결과 버블차트 생성
            bubble_chart_base64 = save_bubble_chart_with_tfidf(video_comments)

            # 감정 분석 결과 HTML 테이블 생성
            sentiment_html = generate_tfidf_sentiment_visualizations(video_comments)

            # 결과를 템플릿에 전달
            context = {
                'wordcloud_image': wordcloud_base64,
                'pie_chart_image': pie_chart_base64,
                'bubble_chart_image': bubble_chart_base64,  # 버블차트 추가
                'sentiment_table': sentiment_html  # TF-IDF 분석 결과 테이블 전달
            }
            return render(request, 'analysis/emotion.html', context)

        except Exception as e:
            print(f"오류 발생: {e}")
            context = {'error_message': f"처리 중 오류가 발생했습니다: {str(e)}"}
            return render(request, 'analysis/emotion.html', context)
    else:
        context = {'error_message': '비디오가 없거나 댓글이 없습니다.'}
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
                    ).exclude(url=video_url)[:10]  # 뉴스 개수 제한
                    
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
                'categorized_news': categorized_news,
                'important_keywords': important_keywords,
                'transcript_segments': transcript_segments
            }
            
        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            context = {
                'section': 'relate',
                'error_message': '분석 중 오류가 발생했습니��.'
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



from bson import ObjectId
from django.shortcuts import get_object_or_404
@login_required
def like_video(request, video_id):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'User not authenticated'}, status=401)

    video_id = ObjectId(video_id)
    video = get_object_or_404(YouTubeData, _id=video_id)
    like_instance, created = Like.objects.get_or_create(user=request.user, youtube_data=video)



    if not created:
        like_instance.delete()  # 이미 찜한 경우 삭제
        return JsonResponse({'message': 'Like removed from like_list', 'is_liked': False})

    return JsonResponse({'message': '해당 기사가 찜 됐어요! 마이페이지에서 확인가능', 'is_liked': True})


# @login_required
def my_liked_videos(request):
    liked_videos = YouTubeData.objects.filter(like__user=request.user)  

    # 각 동영상에 cleaned_title 추가
    processed_videos = []
    for video in liked_videos:
        video.id = str(video._id)
        processed_videos.append({
            "id": video.id,
            "title": video.title,
            "cleaned_title": clean_title(video.title),
            "views": video.views,
            "likes": video.likes,
            "url": video.url,
            "upload_date": video.upload_date,
            "thumbnail": video.thumbnail,
        })

    context = {
        'liked_videos': processed_videos,
        'section': 'mypage'
    }
    return render(request, 'analysis/mypage/my_liked_videos.html', context)

def delete_from_liked(request, id):
    video_id = ObjectId(id)
    video = get_object_or_404(YouTubeData, _id=video_id)
    like_instance = Like.objects.filter(user=request.user, youtube_data=video).first()
    if like_instance:
        like_instance.delete()
    return redirect('account:my_liked_video')


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
from django.contrib import messages

def submit_feedback(request):
    if request.method == 'POST':
        feedback_text = request.POST.get('feedback')
        rating = request.POST.get('rating')

        if feedback_text and rating:
            Feedbacks.objects.create(feedback=feedback_text, rating=int(rating), user=request.user)
            messages.success(request, "피드백이 성공적으로 제출되었습니다!")
            return render(request, 'feedback/feedback_done.html')
        else:
            messages.error(request, "모든 필드를 입력해주세요.")
            return render(request, '/')

def feedback_done(request):
    return render(request, 'feedback/feedback_done.html')

class CustomPasswordChangeView(PasswordChangeView):
    form_class = CustomPasswordChangeForm  # 커스텀 폼 적용
    success_url = reverse_lazy('account:password_change_done')  # 성공 후 이동할 URL
    template_name = 'registration/password_change_form.html'  # 기존 템플릿 경로

@login_required
def update_info(request):
    if request.method == 'POST':
        email = request.POST.get('email', '')
        user = request.user
        if email:
            user.email = email
            user.save()
            messages.success(request, '회원 정보가 성공적으로 수정되었습니다.')
        else:
            messages.error(request, '이메일을 입력해주세요.')
        return redirect('account:mypage')  # 마이페이지로 리다이렉트

    return render(request, 'update_info.html')

@login_required
def password_change(request):
    if request.method == 'POST':
        form = CustomPasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            form.save()  # 비밀번호 저장
            update_session_auth_hash(request, form.user)  # 세션 유지
            return JsonResponse({'message': '비밀번호가 성공적으로 변경되었습니다!'}, status=200)
        else:
            return JsonResponse({'errors': form.errors.as_json()}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)