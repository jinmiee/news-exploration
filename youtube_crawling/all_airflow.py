from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
from googleapiclient.discovery import build
from datetime import datetime, timezone, timedelta
import pytz


class YouTubeDataCollector:
    """
    YouTube 데이터를 수집하고 MongoDB에 저장하는 클래스

    YouTube API를 통해 동영상 데이터 및 댓글 데이터를 수집,
    수집된 데이터를 MongoDB에 저장하는 역할을 수행
    """

    def __init__(self,
                 api_key,
                 mongo_conn_id='mongoid'):
        """
        YouTubeDataCollector 클래스 초기화 메서드.

        Args:
            api_key (str): YouTube API 사용을 위한 API 키.
            mongo_conn_id (str): Airflow에서 설정한 MongoDB 연결 ID.
        """
        self.youtube = build('youtube', 'v3', developerKey=api_key)  # YouTube API 클라이언트를 생성
        self.mongo_hook = MongoHook(conn_id=mongo_conn_id)  # Airflow를 통한 MongoDB 연결 설정
        self.KST = pytz.timezone('Asia/Seoul')  # 한국 표준시 (KST) 타임존 설정

        # 수집 대상 채널의 이름과 ID를 미리 정의한 딕셔너리
        self.channels = {
            "YTN": "UChlgI3UHCOnwUGzWzbJ3H5w",
            "JTBC": "UCsU-I-vHLiaMfV_ceaYz5rQ",
            "KBS": "UCcQTRi69dsVYHN3exePtZ1A",
            "MBC": "UCF4Wxdo3inmxP-Y59wXDsFw",
            "SBS": "UCkinYTS9IHqOEwR1Sze2JTw",
            "연합뉴스": "UCTHCOPwqNfZ0uiKOvFyhGwg"
        }

    def get_video_comments(self, video_id):
        """
        특정 동영상의 댓글을 수집하는 메서드.

        Args:
            video_id (str): YouTube 동영상 ID.

        Returns:
            list: 댓글 내용 포함. 비활성화된 경우 빈 리스트 반환.
        """
        comments = []
        try:
            # YouTube API를 사용해 댓글 데이터를 가져옴
            comment_threads = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100  # 최대 100개의 댓글 수집
            ).execute()

            while comment_threads and len(comments) < 100:  # 댓글 수집이 최대 100개에 도달할 때까지 반복
                for item in comment_threads["items"]:
                    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                    comments.append({"author": author, "comment": comment})

                    if len(comments) >= 100:  # 100개를 초과하면 종료
                        break

                # 다음 페이지가 있으면 계속 수집
                if "nextPageToken" in comment_threads:
                    comment_threads = self.youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        textFormat="plainText",
                        pageToken=comment_threads["nextPageToken"],
                        maxResults=100
                    ).execute()
                else:
                    break
        except Exception as e:
            # 댓글 비활성화 에러 처리
            if "commentsDisabled" in str(e):
                print(f"댓글 비활성화된 동영상: {video_id}")
            else:
                print(f"댓글 수집 오류: {e}")
        return comments

    def collect_channel_data(self, channel_name, channel_id):
        """
        채널 데이터 수집 함수
        """
        try:
            now_kst = datetime.now(self.KST)  # 현재 시간 (KST)
            start_time_kst = now_kst - timedelta(hours=12)  # 12시간 전 (KST)
            start_time_utc = start_time_kst.astimezone(pytz.utc)  # UTC 시간으로 변환
            start_time = start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')  # YouTube API에 필요한 ISO 8601 형식

            videos = []
            page_token = None

            while True:
                # YouTube API 호출: 지정된 채널의 동영상 목록을 가져옴
                search_response = self.youtube.search().list(
                    order='date',  # 최신순 정렬
                    part='snippet',  # 메타데이터 포함
                    channelId=channel_id,  # 채널 ID
                    maxResults=50,  # 최대 50개 결과
                    publishedAfter=start_time,  # 12시간 전부터 데이터 가져오기
                    type='video',  # 동영상만 포함
                    pageToken=page_token
                ).execute()

                video_ids = [item['id']['videoId'] for item in search_response['items']]

                videos_response = self.youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(video_ids)
                ).execute()

                for video in videos_response['items']:
                    # 실시간 동영상 건너뛰기
                    if video['snippet'].get('liveBroadcastContent') != 'none':
                        print(f"라이브 동영상 건너뜀: {video['id']}")
                        continue

                    # 댓글이 비활성화된 경우 건너뛰기
                    comments = self.get_video_comments(video['id'])
                    if not comments:  # 댓글이 비활성화된 경우
                        print(f"댓글 비활성화된 동영상 건너뜀: {video['id']}")
                        continue

                    # 동영상 데이터 생성
                    published_at = video['snippet']['publishedAt']
                    published_at_kst = datetime.strptime(
                        published_at, '%Y-%m-%dT%H:%M:%SZ'
                    ).replace(tzinfo=timezone.utc).astimezone(self.KST)

                    video_data = {
                        'channel_name': channel_name,
                        'title': video['snippet'].get('title'),
                        'views': video['statistics'].get('viewCount', '0'),
                        'upload_date': published_at_kst,
                        'url': f"https://www.youtube.com/watch?v={video['id']}",
                        'channel': video['snippet'].get('channelTitle'),
                        'desc': video['snippet'].get('description', ''),
                        'likes': video['statistics'].get('likeCount', '0'),
                        'comments': comments,
                        'thumbnail': video['snippet']['thumbnails']['high']['url']
                    }
                    videos.append(video_data)

                # 다음 페이지로 이동
                if 'nextPageToken' in search_response and len(videos) < 100:
                    page_token = search_response['nextPageToken']
                else:
                    break

            # MongoDB 저장
            if videos:
                mongo_client = self.mongo_hook.get_conn()
                db = mongo_client['youtube_data']
                collection = db['youtube_datas']

                for video in videos:
                    collection.update_one(
                        {'url': video['url']},
                        {'$set': video},
                        upsert=True
                    )
                    print(f"{channel_name} - 데이터 저장 또는 업데이트: {video['title']}")

        except Exception as e:
            print(f"{channel_name} 데이터 수집 오류: {e}")
    
    def collect_all_channel_data(self):
        """
        모든 채널의 데이터를 순차적으로 수집하는 메서드.
        """
        for name, channel_id in self.channels.items():
            self.collect_channel_data(name, channel_id)  # 각 채널 데이터 수집 호출


# Airflow DAG 기본 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 18),  # 시작 날짜
    'retries': 1,  # 실패 시 재시도 횟수
    'retry_delay': timedelta(minutes=5),  # 재시도 간격
}

with DAG(
    'youtube_data_collection_dag',  # DAG 이름
    default_args=default_args,  # 기본 설정
    description='YouTube Data Collection DAG',
    schedule_interval='0 11,23 * * *',  # 매일 11:00, 23:00 실행 
    catchup=False,  # 이전 날짜의 DAG 실행 방지
    max_active_runs=1  # 병렬 실행 방지
) as dag:
    def run_youtube_data_collection(api_key):
        """
        YouTube 데이터 수집 작업 실행 함수.

        Args:
            api_key (str): YouTube API 키.
        """
        collector = YouTubeDataCollector(api_key=api_key)  # 데이터 수집기 초기화
        collector.collect_all_channel_data()  # 모든 채널 데이터 수집 호출

    # PythonOperator로 DAG 작업 정의
    collect_youtube_data = PythonOperator(
        task_id='collect_youtube_data',
        python_callable=lambda: run_youtube_data_collection('AIzaSyCDFmk4W1Z-hB3u5UXGDDgfngEb148ZgIc'),  # API 키 전달
        dag=dag
    )
    collect_youtube_data