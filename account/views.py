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


# @login_required
def chart(request): # 차트 뷰
    # 현재 시간 가져오기
    now = timezone.now()
    # 오늘 오후 6시 (KST) 계산
    today_6pm_kst = timezone.localtime().replace(hour=18, minute=0, second=0, microsecond=0)
    # 어제 오후 6시 (KST) 계산
    yesterday_6pm_kst = today_6pm_kst - timedelta(days=1)

    # KST를 UTC로 변환 (UTC = KST-9)
    yesterday_6pm_utc = yesterday_6pm_kst - timedelta(hours=9)
    today_6pm_utc = today_6pm_kst - timedelta(hours=9)

    # 쿼리 실행: 어제 오후 6시 ~ 오늘 오후 6시 사이에 업로드 된 동영상 가져오기
    # 조회수 기준 내림차순 정렬, 상위 10개 데이터 가져오기
    top_news = YouTubeData.objects.filter(
        upload_date__gte=yesterday_6pm_utc, # 어제 오후 6시 이후 데이터
        upload_date__lte=today_6pm_utc  # 오늘 오후 6시 이전 데이터
    ).order_by('-views')[:10]


    # 템플릿에 데이터 전달
    return render(request, 'analysis/chart.html', {'top_news': top_news})

# 동영상 세부 정보 조회 API
@csrf_exempt # CSRF 검사 비활성화(POST 요청 허용)
def video_details(request):
    if request.method == 'POST':
        data = json.loads(request.body) # 요청 본문에서 JSON 데이터 로드
        video_url = data.get('url') # JSON 데이터에서 URL 추출

        # 데이터베이스에서 URL로 YouTubeData 검색
        video = YouTubeData.objects.filter(url=video_url).first()

        # 검색된 데이터가 없을 경우 에러 응답
        if not video:
            return JsonResponse({"error": "Video not found."}, status=404)

        # 검색된 데이터가 있을 경우 URL 정보를 JSON 응답으로 반환
        return JsonResponse({"video_url": video.url})
    
    # 요청이 POST 방식이 아닐 경우 에러 응답
    return JsonResponse({"error": "Invalid request method."}, status=400)


<<<<<<< HEAD
#상세분석
# @login_required
def detail(request):
    # 현재 시간과 어제 오후 6시 및 오늘 오후 6시 계산
    now = timezone.now()
    today_6pm_kst = timezone.localtime().replace(hour=18, minute=0, second=0, microsecond=0)
    yesterday_6pm_kst = today_6pm_kst - timedelta(days=1)

    # KST를 UTC로 변환
    yesterday_6pm_utc = yesterday_6pm_kst - timedelta(hours=9)
    today_6pm_utc = today_6pm_kst - timedelta(hours=9)

    # 쿼리 실행: 어제 오후 6시 ~ 오늘 오후 6시 데이터 가져오기
    top_news = YouTubeData.objects.filter(
        upload_date__gte=yesterday_6pm_utc,
        upload_date__lte=today_6pm_utc
    ).order_by('-views')[:10]

    first_item = top_news[1]

    # URL 파싱
    parsed_url = urlparse(first_item.url)

    # 쿼리 파라미터 추출
    query_params = parse_qs(parsed_url.query)

    # video_id 추출
    video_id = query_params.get('v', [None])[0]

    context = {
        'news': first_item,
        'section': 'detail',
        'video_id': video_id
    }

    return render(request, 'analysis/detail.html', context)
=======
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
>>>>>>> 3c78d531bf4f4147553b8eecb5e721211fc47592



# @login_required
def emotion(request):
    return render(request, 'analysis/emotion.html', {'section': 'emotion'}) 

# @login_required
def relate(request):
    return render(request, 'analysis/relate.html', {'section': 'relate'}) 

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