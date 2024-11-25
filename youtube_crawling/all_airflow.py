from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
from googleapiclient.discovery import build
from datetime import datetime, timezone, timedelta
import pytz


# YouTube 데이터 수집 클래스 정의
class YouTubeDataCollector:
    def __init__(self,
                 api_key,  # API 키를 직접 전달
                 mongo_conn_id='mongoid'):  # Airflow에서 정의된 MongoDB 연결 ID
        """
        YouTubeDataCollector 클래스 초기화 메서드.

        YouTube API를 통해 동영상 데이터 및 댓글 데이터를 수집하고,
        MongoDB에 저장하는 역할을 하는 클래스입니다.

        Args:
            api_key (str): YouTube API 사용을 위한 API 키.
            mongo_conn_id (str): Airflow에서 설정한 MongoDB 연결 ID.
        """
        # YouTube API 클라이언트 생성 (API 키를 통해 YouTube API 연결)
        self.youtube = build('youtube', 'v3', developerKey=api_key)

        # MongoDB 연결을 위한 MongoHook 생성
        self.mongo_hook = MongoHook(conn_id=mongo_conn_id)

        # 한국 표준시(KST) 시간대를 설정
        self.KST = pytz.timezone('Asia/Seoul')

        # 현재 날짜와 시간을 기준으로 MongoDB에 저장할 데이터의 이름을 설정 (예: 1120_18)
        self.today_date = datetime.now(self.KST).strftime('%m%d_%H')

        # YouTube 채널 ID 목록
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

        주어진 video_id에 해당하는 YouTube 동영상의 댓글을 최대 100개까지 수집하여 반환합니다.

        Args:
            video_id (str): YouTube 동영상 ID.

        Returns:
            list: 댓글 목록 (작성자, 댓글 내용 포함).
        """
        comments = []  # 수집된 댓글을 저장할 리스트
        try:
            # 동영상 댓글 스레드 요청 (최대 100개 댓글 요청)
            comment_threads = self.youtube.commentThreads().list(
                part="snippet",  # 댓글 데이터에서 가져올 필드
                videoId=video_id,  # 대상 동영상 ID
                textFormat="plainText",  # 댓글 내용을 일반 텍스트로 반환
                maxResults=100  # 최대 100개의 댓글 요청
            ).execute()

            # 댓글 데이터 처리
            while comment_threads and len(comments) < 100:
                for item in comment_threads["items"]:
                    # 댓글 작성자 및 댓글 내용 추출
                    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                    comments.append({"author": author, "comment": comment})

                    # 수집한 댓글이 100개를 넘으면 종료
                    if len(comments) >= 100:
                        break

                # 다음 페이지의 댓글 요청 (있으면 계속해서 댓글을 더 수집)
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
            print(f"댓글 수집 오류: {e}")  # 예외 발생 시 에러 메시지 출력
        return comments

    def collect_channel_data(self, channel_name, channel_id):
        """
        특정 채널의 최근 24시간 동안의 YouTube 데이터를 수집하는 메서드.

        주어진 channel_id에 해당하는 YouTube 채널에서 최근 24시간 동안 업로드된 동영상을 수집하고,
        관련 데이터를 MongoDB에 저장합니다.

        Args:
            channel_name (str): 채널 이름.
            channel_id (str): YouTube 채널 ID.
        """
        # 현재 시간 및 24시간 전 시간을 KST(한국 시간대) 기준으로 계산
        KST = pytz.timezone('Asia/Seoul')
        now_kst = datetime.now(KST)
        yesterday_kst = now_kst - timedelta(hours=24)
        yesterday_utc = yesterday_kst.astimezone(pytz.utc)
        start_time = yesterday_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

        # MongoDB 컬렉션 이름 생성 (현재 날짜/시간 기반)
        collection_name = f"youtube_data_{self.today_date}"

        videos = []  # 수집된 동영상 데이터를 저장할 리스트
        page_token = None  # YouTube API의 페이지 토큰 초기화

        try:
            # 24시간 이내 업로드된 동영상 검색 (최대 50개씩 요청)
            while True:
                search_response = self.youtube.search().list(
                    order='date',
                    part='snippet',
                    channelId=channel_id,
                    maxResults=50,
                    publishedAfter=start_time,  # 여기에 RFC 3339 형식의 타임스탬프 사용
                    type='video',
                    pageToken=page_token
                ).execute()

                # 동영상 ID 추출
                video_ids = [item['id']['videoId'] for item in search_response['items']]

                # 동영상 상세 정보 요청
                videos_response = self.youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(video_ids)  # 요청 ID를 쉼표로 구분하여 전달
                ).execute()

                # 각 동영상의 데이터 처리
                for video in videos_response['items']:
                    if video['snippet'].get('liveBroadcastContent') == 'none':  # 라이브 방송 제외
                        # 동영상 발행 시간을 KST로 변환
                        published_at = video['snippet']['publishedAt']
                        published_at_kst = datetime.strptime(
                            published_at, '%Y-%m-%dT%H:%M:%SZ'
                        ).replace(tzinfo=timezone.utc).astimezone(self.KST)

                        # 24시간 내 업로드된 동영상만 처리
                        if published_at_kst >= yesterday_kst:
                            video_data = {
                                'channel_name': channel_name,
                                'title': video['snippet'].get('title'),
                                'views': video['statistics'].get('viewCount', '0'),
                                'upload_date': published_at_kst.strftime('%Y-%m-%d %H:%M:%S %Z'),
                                'url': f"https://www.youtube.com/watch?v={video['id']}",
                                'channel': video['snippet'].get('channelTitle'),
                                'desc': video['snippet'].get('description', ''),
                                'likes': video['statistics'].get('likeCount', '0'),
                                'comments': self.get_video_comments(video['id'])  # 댓글 수집
                            }
                            videos.append(video_data)

                # 다음 페이지 요청 (다음 동영상이 있으면 계속해서 요청)
                if 'nextPageToken' in search_response and len(videos) < 100:
                    page_token = search_response['nextPageToken']
                else:
                    break

            # 조회수 기준 상위 20개 동영상 선택
            videos.sort(key=lambda x: int(x['views']), reverse=True)
            videos = videos[:20]  # 상위 20개 동영상만 선택

            # MongoDB에 데이터 저장
            if videos:
                mongo_client = self.mongo_hook.get_conn()
                db = mongo_client['youtube_data']  # 'youtube_data' DB
                collection = db[collection_name]  # 동영상 데이터를 저장할 컬렉션
                collection.insert_many(videos)  # 데이터 삽입
                print(f"{channel_name} 데이터 저장 완료 (상위 20개)")  # 저장 완료 메시지 출력

        except Exception as e:
            print(f"{channel_name} 데이터 수집 오류: {e}")  # 예외 발생 시 에러 메시지 출력

    def collect_all_channel_data(self):
        """
        모든 채널의 데이터를 순차적으로 수집하는 메서드.

        모든 채널의 데이터를 한 번에 수집하여 MongoDB에 저장합니다.
        """
        for name, channel_id in self.channels.items():
            self.collect_channel_data(name, channel_id)  # 각 채널 데이터 수집


# Airflow DAG 기본 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 18),  # 시작 날짜
    'retries': 1,  # 실패 시 재시도 횟수
    'retry_delay': timedelta(minutes=5),  # 재시도 간격
}

# Airflow DAG 생성
with DAG(
        'youtube_data_collection_dag',
        default_args=default_args,
        description='YouTube Data Collection DAG',
        schedule_interval='0 18 * * *',  # 매일 오후 6시(18:00)에 실행
        catchup=False  # 이전 실행을 건너뛰고 현재 실행만 진행
) as dag:
    def run_youtube_data_collection(api_key):
        """
        YouTube 데이터 수집 작업 실행 함수.

        API 키를 받아 YouTubeDataCollector 인스턴스를 생성하고, 데이터를 수집합니다.

        Args:
            api_key (str): YouTube API 키.
        """
        collector = YouTubeDataCollector(api_key=api_key)  # YouTubeDataCollector 인스턴스 생성
        collector.collect_all_channel_data()  # 모든 채널 데이터 수집


    # Airflow에서 직접 API 키를 입력하도록 수정
    collect_youtube_data = PythonOperator(
        task_id='collect_youtube_data',
        python_callable=lambda: run_youtube_data_collection('AIzaSyC9_BEfIDVhNZgeoAkVAV2P5YgEGQW7YTs'),  # 직접 API 키 입력
        dag=dag
    )

    collect_youtube_data  # DAG의 태스크 실행