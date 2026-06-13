from googleapiclient.discovery import build
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytz
import os
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()

# MongoDB 설정
client = MongoClient(os.getenv("MONGO_URI"))
db = client['youtube_data']

# YouTube API 설정
API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# 시간 설정
KST = pytz.timezone('Asia/Seoul')
today_date = datetime.now(KST).strftime('%m%d_%H')

# 채널 ID 리스트
channels = {
    "YTN": "UChlgI3UHCOnwUGzWzbJ3H5w",
    "JTBC": "UCsU-I-vHLiaMfV_ceaYz5rQ",
    "KBS": "UCcQTRi69dsVYHN3exePtZ1A",
    "MBC": "UCF4Wxdo3inmxP-Y59wXDsFw",
    "SBS": "UCkinYTS9IHqOEwR1Sze2JTw",
    "연합뉴스": "UCTHCOPwqNfZ0uiKOvFyhGwg"
}

def get_video_comments(video_id):
    comments = []
    try:
        comment_threads = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        ).execute()

        while comment_threads and len(comments) < 100:
            for item in comment_threads["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                comments.append({"author": author, "comment": comment})
                if len(comments) >= 100:
                    break

            if "nextPageToken" in comment_threads:
                comment_threads = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    pageToken=comment_threads["nextPageToken"],
                    maxResults=100
                ).execute()
            else:
                break
    except Exception as e:
        logger.error(f"댓글 수집 오류: {e}")
    return comments

def set_video(channel_name, channel_id):
    now_kst = datetime.now(KST)
    yesterday_kst = now_kst - timedelta(hours=24)
    yesterday_utc = yesterday_kst.astimezone(timezone.utc).isoformat()

    collection_name = f"youtube_data_{today_date}"  # 모든 데이터를 하나의 컬렉션에 저장
    collection = db[collection_name]

    videos = []
    page_token = None

    try:
        while True:
            search_response = youtube.search().list(
                order='date',
                part='snippet',
                channelId=channel_id,
                maxResults=50,
                publishedAfter=yesterday_utc,
                type='video',
                pageToken=page_token
            ).execute()

            video_ids = [item['id']['videoId'] for item in search_response['items']]
            videos_response = youtube.videos().list(
                part='snippet,statistics',
                id=','.join(video_ids)
            ).execute()

            for video in videos_response['items']:
                if video['snippet'].get('liveBroadcastContent') == 'none':
                    published_at = video['snippet']['publishedAt']
                    published_at_kst = datetime.strptime(
                        published_at, '%Y-%m-%dT%H:%M:%SZ'
                    ).replace(tzinfo=timezone.utc).astimezone(KST)

                    if published_at_kst >= yesterday_kst:
                        video_data = {
                            'channel_name': channel_name,  # 채널 이름 추가
                            'title': video['snippet'].get('title'),
                            'views': video['statistics'].get('viewCount', '0'),
                            'upload_date': published_at_kst.strftime('%Y-%m-%d %H:%M:%S %Z'),
                            'url': f"https://www.youtube.com/watch?v={video['id']}",
                            'channel': video['snippet'].get('channelTitle'),
                            'desc': video['snippet'].get('description', ''),
                            'likes': video['statistics'].get('likeCount', '0'),
                            'comments': get_video_comments(video['id']),
                            'thumbnail': video['snippet']['thumbnails']['high']['url']  # 썸네일 URL
                        }
                        videos.append(video_data)

            if 'nextPageToken' in search_response and len(videos) < 100:  # 100개 이상 가져오지 않음
                page_token = search_response['nextPageToken']
            else:
                break

        # 조회수 기준 상위 20개 선택
        videos.sort(key=lambda x: int(x['views']), reverse=True)
        videos = videos[:20]

        if videos:
            collection.insert_many(videos)
            logger.info(f"{channel_name} 데이터 저장 완료 (상위 20개)")
    except Exception as e:
        logger.error(f"{channel_name} 데이터 수집 오류: {e}")

# 모든 채널 데이터 수집 실행
for name, channel_id in channels.items():
    set_video(name, channel_id)
