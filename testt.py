'''
pip install konlpy networkx matplotlib pandas
'''
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'news.settings')
django.setup()
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


from account.forms import UserRegistrationForm
from account.models import YouTubeData, Like
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



from account.analysis.clustering import choose_10


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

# 데이터 필터링
all_news = YouTubeData.objects.filter(
    upload_date__gte=analysis_start, upload_date__lte=analysis_end)

titles = [news.title for news in all_news]
views = [news.views for news in all_news]
ids = [news._id for news in all_news]

choose_10(titles,views,ids)