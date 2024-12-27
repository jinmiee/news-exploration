"""
URL configuration for news project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views
from django.urls import reverse_lazy

app_name = 'account'

urlpatterns = [
    
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('password-change/', views.CustomPasswordChangeView.as_view(), name='password_change'),    path('password-change/done/', auth_views.PasswordChangeDoneView.as_view(), name='password_change_done'),
    
    path('', views.chart, name='chart'),
    path('video-details/', views.video_details, name='video_details'),
    path('detail/<str:video_id>/', views.detail, name='detail_by_id'),  # ID를 통한 접근
    path('detail/', views.detail, name='detail_by_url'),  # URL을 통한 접근
    path('feedback/', views.submit_feedback, name='submit_feedback'),
    path('feedback_done/', views.feedback_done, name='feedback_done'),

    path('weekly-issues/', views.weekly_issues, name='weekly_issues'),
    path('emotion/', views.emotion, name='emotion'),
    path('relate/', views.relate, name='relate'),
    path('register/', views.register, name='register'),
    path('find-username/', views.find_username, name='find_username'),
    path('mypage/', views.mypage, name='mypage'),
    path('like/<str:video_id>/', views.like_video, name='like_video'),
    path('mypage/like/', views.my_liked_videos, name='my_liked_video'),
    path('mypage/like/<str:id>/', views.delete_from_liked, name='delete_from_liked'),

    path('related-duplicates/', views.get_related_duplicate_videos, name='related_duplicates'),


    path('signup/', views.register, name='signup'),
    path('find-password/', views.find_password, name='find_password'),

    path('update-info/', views.update_info, name='update_info'),  # 새 URL 추가
path('password_change/', views.password_change, name='password_change'),
]
