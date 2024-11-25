from datetime import timedelta

from django.http import JsonResponse
from django.utils import timezone
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
import json

from .models import YouTubeData

# @login_required
def chart(request):
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

    # 템플릿에 데이터 전달
    return render(request, 'analysis/chart.html', {'top_news': top_news})

@csrf_exempt
def video_details(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        video_url = data.get('url')

        # URL로 데이터 검색
        video = YouTubeData.objects.filter(url=video_url).first()

        if not video:
            return JsonResponse({"error": "Video not found."}, status=404)

        # 댓글 가져오기
        comments = [{"author": c['author'], "comment": c['comment']} for c in video.comments]

        return JsonResponse({"video_url": video.url, "comments": comments})

    return JsonResponse({"error": "Invalid request method."}, status=400)

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