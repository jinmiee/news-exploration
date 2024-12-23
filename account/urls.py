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
    path('password-change/', auth_views.PasswordChangeView.as_view(success_url=reverse_lazy('account:password_change_done')), name='password_change'),
    path('password-change/done/', auth_views.PasswordChangeDoneView.as_view(), name='password_change_done'),
    
    path('', views.chart, name='chart'),
    path('video-details/', views.video_details, name='video_details'),
    path('detail/', views.detail, name='detail'),
    path('weekly-issues/', views.weekly_issues, name='weekly_issues'),
    path('emotion/', views.emotion, name='emotion'),
    path('relate/', views.relate, name='relate'),
    path('register/', views.register, name='register'),
    path('find-username/', views.find_username, name='find_username'),
    path('mypage/', views.mypage, name='mypage'),
    path('like/<str:video_id>/', views.like_video, name='like_video'),
    path('mypage/like/', views.my_liked_videos, name='my_liked_video'),



    path('signup/', views.register, name='signup'),
    path('find-password/', views.find_password, name='find_password'),
    path('analysis/get_performance_metrics/', views.get_performance_metrics, name='get_performance_metrics'),

]
