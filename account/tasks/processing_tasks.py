
from collections import defaultdict
from datetime import timedelta, datetime

from django.utils import timezone

from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer

from pytz import timezone
import pytz

import matplotlib
matplotlib.use('Agg')
from django.utils.timezone import localtime, utc, now

from sklearn.metrics.pairwise import cosine_similarity
from account.analysis.text_processing import clean_title, process_text, get_top10_chart_based, get_hybrid_similarity,get_bert_similarity_batch

from django.utils.timezone import localtime
from datetime import timedelta
from account.models import Like, YouTubeData, WeeklyIssue, Chart, WeeklyIssueDuplicateVideo, ChartDuplicateVideo

def save_top_videos(start_time, end_time, model):
    """
    특정 시간대의 상위 10개 동영상을 지정된 모델에 저장
    :param start_time: 데이터 필터링 시작 시간
    :param end_time: 데이터 필터링 종료 시간
    :param model: 데이터를 저장할 Django 모델 (Chart, WeeklyIssue 등)
    """
    try:
        # 데이터베이스에서 시간 범위에 해당하는 데이터 가져오기
        print(f"save_top_videos 실행됨: {start_time} ~ {end_time}")  # 디버깅 로그
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
        expired_charts = Chart.objects.filter(upload_date__lt=expiration_time)
        deleted_count, _ = expired_charts.delete()

        logger.info(f"{deleted_count}개의 24시간 지난 Chart 데이터가 삭제되었습니다.")
    except Exception as e:
        logger.error(f"delete_expired_charts 실행 중 오류 발생: {e}")

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

def extract_duplicates_for_weekly_issues():
    """
    주간 이슈 데이터를 기준으로 중복 동영상을 추출하고 WeeklyIssueDuplicateVideo에 저장.
    """
    try:
        print("STEP 1: 주간 이슈 및 전체 동영상 데이터 가져오기")
        weekly_videos = list(WeeklyIssue.objects.all())
        all_videos = list(YouTubeData.objects.all())
        print(f"DEBUG: 주간 이슈 비디오 개수: {len(weekly_videos)}, 전체 유튜브 데이터 개수: {len(all_videos)}")

        print("STEP 2: 텍스트 전처리")
        weekly_corpus = [process_text(video.title, video.transcript or []) for video in weekly_videos]
        all_corpus = [process_text(video.title, video.transcript or []) for video in all_videos]
        print("DEBUG: 텍스트 전처리 완료")

        if not weekly_corpus or not all_corpus:
            print("Error: 주간 이슈 또는 전체 유튜브 데이터가 비어 있습니다.")
            return

        print("STEP 3: TF-IDF 계산")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        weekly_tfidf_matrix = vectorizer.fit_transform(weekly_corpus)
        all_tfidf_matrix = vectorizer.transform(all_corpus)
        print("DEBUG: TF-IDF 계산 완료")

        print("STEP 4: KoBERT 유사도 계산")
        weekly_kobert_similarity = get_bert_similarity_batch(weekly_corpus, batch_size=32)
        all_kobert_similarity = get_bert_similarity_batch(all_corpus, batch_size=32)
        print("DEBUG: KoBERT 유사도 계산 완료")

        print("STEP 5: Hybrid 유사도 계산")
        weekly_hybrid_similarity = get_hybrid_similarity(
            weekly_tfidf_matrix, weekly_kobert_similarity,
            weight_tfidf=0.6, weight_bert=0.4
        )
        all_hybrid_similarity = get_hybrid_similarity(
            all_tfidf_matrix, all_kobert_similarity,
            weight_tfidf=0.6, weight_bert=0.4
        )
        print("DEBUG: Hybrid 유사도 계산 완료")

        print("STEP 6: 중복 탐지")
        threshold = 0.6  # 유사도 임계값
        duplicates = set()

        for i, all_video in enumerate(all_videos):
            for j, weekly_video in enumerate(weekly_videos):
                similarity = all_hybrid_similarity[i, j]

                # URL 또는 높은 유사도를 기반으로 중복 판단
                if all_video.url == weekly_video.url or similarity > threshold:
                    duplicates.add(all_video._id)

        print(f"DEBUG: 총 {len(duplicates)}개의 중복 비디오 탐지 완료")

        print("STEP 7: 중복 동영상 저장")
        for duplicate_id in duplicates:
            duplicate_video = YouTubeData.objects.get(_id=duplicate_id)
            WeeklyIssueDuplicateVideo.objects.update_or_create(
                _id=duplicate_video._id,
                defaults={
                    "title": duplicate_video.title,
                    "url": duplicate_video.url,
                    "views": duplicate_video.views,
                    "upload_date": duplicate_video.upload_date,
                    "channel_name": duplicate_video.channel_name,
                    "thumbnail": duplicate_video.thumbnail,
                    "likes": duplicate_video.likes,
                    "transcript": duplicate_video.transcript,
                }
            )
            print(f"DEBUG: 중복 비디오 저장 완료 - 제목: {duplicate_video.title}, URL: {duplicate_video.url}")

        print(f"주간 이슈 기준 {len(duplicates)}개의 중복 동영상이 저장되었습니다.")
    except Exception as e:
        print(f"주간 이슈 중복 동영상 추출 오류: {e}")


def extract_duplicates_for_chart():
    """
    차트 데이터를 기준으로 중복 동영상을 추출하고 ChartDuplicateVideo에 저장.
    """
    try:
        print("STEP 1: 차트 데이터와 최근 3일 이내의 전체 동영상 데이터 가져오기")

        # 1. 차트 데이터 가져오기
        chart_videos = list(Chart.objects.all())
        print(f"DEBUG: 차트 비디오 개수: {len(chart_videos)}")

        # 2. 최근 3일 이내 동영상 필터링
        current_time = now()
        three_days_ago = current_time - timedelta(days=3)
        recent_videos = list(YouTubeData.objects.filter(upload_date__gte=three_days_ago, upload_date__lte=current_time))
        print(f"DEBUG: 최근 3일 이내 전체 동영상 개수: {len(recent_videos)}")

        # 3. 텍스트 전처리
        print("STEP 2: 텍스트 전처리")
        chart_corpus = [process_text(video.title, video.transcript or []) for video in chart_videos]
        recent_corpus = [process_text(video.title, video.transcript or []) for video in recent_videos]
        print("DEBUG: 텍스트 전처리 완료")

        if not chart_corpus or not recent_corpus:
            print("Error: 차트 데이터 또는 최근 동영상 데이터가 비어 있습니다.")
            return

        # 4. TF-IDF 계산
        print("STEP 3: TF-IDF 계산")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        chart_tfidf_matrix = vectorizer.fit_transform(chart_corpus)
        recent_tfidf_matrix = vectorizer.transform(recent_corpus)
        print("DEBUG: TF-IDF 계산 완료")

        # 5. KoBERT 유사도 계산
        print("STEP 4: KoBERT 유사도 계산")
        chart_kobert_similarity = get_bert_similarity_batch(chart_corpus, batch_size=32)
        recent_kobert_similarity = get_bert_similarity_batch(recent_corpus, batch_size=32)
        print("DEBUG: KoBERT 유사도 계산 완료")

        # 6. Hybrid 유사도 계산
        print("STEP 5: Hybrid 유사도 계산")
        hybrid_similarity = get_hybrid_similarity(
            recent_tfidf_matrix, recent_kobert_similarity,
            weight_tfidf=0.6, weight_bert=0.4
        )
        print("DEBUG: Hybrid 유사도 계산 완료")

        # 7. 중복 탐지
        print("STEP 6: 중복 탐지")
        threshold = 0.6  # 유사도 임계값
        duplicates = set()

        for i, recent_video in enumerate(recent_videos):
            for j, chart_video in enumerate(chart_videos):
                similarity = hybrid_similarity[i, j]

                # URL 또는 높은 유사도를 기반으로 중복 판단
                if recent_video.url == chart_video.url or similarity > threshold:
                    duplicates.add(recent_video._id)

        print(f"DEBUG: 총 {len(duplicates)}개의 중복 비디오 탐지 완료")

        # 8. 중복 동영상 저장
        print("STEP 7: 중복 동영상 저장")
        for duplicate_id in duplicates:
            duplicate_video = YouTubeData.objects.get(_id=duplicate_id)
            ChartDuplicateVideo.objects.update_or_create(
                _id=duplicate_video._id,
                defaults={
                    "title": duplicate_video.title,
                    "url": duplicate_video.url,
                    "views": duplicate_video.views,
                    "upload_date": duplicate_video.upload_date,
                    "channel_name": duplicate_video.channel_name,
                    "thumbnail": duplicate_video.thumbnail,
                    "likes": duplicate_video.likes,
                    "transcript": duplicate_video.transcript,
                }
            )
            print(f"DEBUG: 중복 비디오 저장 완료 - 제목: {duplicate_video.title}, URL: {duplicate_video.url}")

        print(f"차트 기준 {len(duplicates)}개의 중복 동영상이 저장되었습니다.")
    except Exception as e:
        print(f"차트 중복 동영상 추출 오류: {e}")


