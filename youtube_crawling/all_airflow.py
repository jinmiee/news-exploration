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
            list: 댓글 내용 포함.
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
                    # 댓글 내용을 리스트에 추가
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
        # 현재 한국 시간(KST)과 어제 시간을 계산
        now_kst = datetime.now(self.KST)  # 현재 시간 (KST)
        yesterday_kst = now_kst - timedelta(hours=24)  # 24시간 전 (KST)
        yesterday_utc = yesterday_kst.astimezone(pytz.utc)  # UTC 시간으로 변환
        start_time = yesterday_utc.strftime('%Y-%m-%dT%H:%M:%SZ')  # YouTube API에 필요한 ISO 8601 형식

        videos = []  # 수집된 동영상 데이터를 저장할 리스트
        page_token = None  # YouTube API의 페이지 토큰 (다음 페이지로 이동 시 필요)

        try:
            while True:
                # YouTube API 호출: 지정된 채널의 동영상 목록을 가져옴
                search_response = self.youtube.search().list(
                    order='date',  # 최신 순으로 정렬
                    part='snippet',  # 메타데이터 정보 가져오기
                    channelId=channel_id,  # 채널 ID
                    maxResults=50,  # 한 번에 가져올 최대 동영상 수
                    publishedAfter=start_time,  # 24시간 이전 동영상은 제외
                    type='video',  # 동영상만 필터링
                    pageToken=page_token  # 페이지 토큰
                ).execute()

                # 각 동영상의 ID를 추출
                video_ids = [item['id']['videoId'] for item in search_response['items']]

                # 동영상 상세 정보 가져오기
                videos_response = self.youtube.videos().list(
                    part='snippet,statistics',  # 제목, 설명, 통계 정보 등 포함
                    id=','.join(video_ids)  # 여러 동영상 ID를 쉼표로 구분하여 요청
                ).execute()

                # 각 동영상 데이터 처리
                for video in videos_response['items']:
                    # 실시간 동영상(liveBroadcastContent)이 아닌 경우만 처리
                    if video['snippet'].get('liveBroadcastContent') == 'none':
                        # 동영상 업로드 시간을 UTC 기준으로 파싱 후 KST로 변환
                        published_at = video['snippet']['publishedAt']
                        published_at_kst = datetime.strptime(
                            published_at, '%Y-%m-%dT%H:%M:%SZ'
                        ).replace(tzinfo=timezone.utc).astimezone(self.KST)

                        # 업로드 시간이 어제 이후인 동영상만 처리
                        if published_at_kst >= yesterday_kst:
                            # 댓글 수집 시 비활성화된 동영상 건너뛰기
                            comments = self.get_video_comments(video['id'])
                            if comments == []:  # 댓글 비활성화인 경우
                                continue

                            # 동영상 데이터 생성
                            video_data = {
                                'channel_name': channel_name,  # 채널 이름
                                'title': video['snippet'].get('title'),  # 동영상 제목
                                'views': video['statistics'].get('viewCount', '0'),  # 조회수
                                'upload_date': published_at_kst,  # 업로드 날짜 (datetime 객체)
                                'url': f"https://www.youtube.com/watch?v={video['id']}",  # 동영상 URL
                                'channel': video['snippet'].get('channelTitle'),  # 채널 제목
                                'desc': video['snippet'].get('description', ''),  # 동영상 설명
                                'likes': video['statistics'].get('likeCount', '0'),  # 좋아요 수
                                'comments': self.get_video_comments(video['id']),  # 댓글 목록
                                'thumbnail': video['snippet']['thumbnails']['high']['url']  # 썸네일 URL
                            }
                            videos.append(video_data)  # 리스트에 추가

                # 다음 페이지가 있으면 페이지 토큰 갱신, 없으면 종료
                if 'nextPageToken' in search_response and len(videos) < 100:
                    page_token = search_response['nextPageToken']
                else:
                    break

            # 조회수 기준으로 정렬 후 상위 20개 선택
            videos.sort(key=lambda x: int(x['views']), reverse=True)
            videos = videos[:20]

            # MongoDB에 데이터 저장
            if videos:
                mongo_client = self.mongo_hook.get_conn()  # MongoDB 연결
                db = mongo_client['youtube_data']  # 데이터베이스 선택
                collection = db['youtube_datas']  # 컬렉션 선택

                for video in videos:
                    # 중복 확인: URL 기준
                    if not collection.find_one({'url': video['url']}):
                        collection.insert_one(video)
                        print(f"{channel_name} - 새 데이터 저장: {video['title']}")
                    else:
                        print(f"{channel_name} - 중복 데이터 건너뜀: {video['title']}")

                    # 데이터 저장 완료 메시지 추가
                print(f"{channel_name} 데이터 저장 완료 (상위 20개)")

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
    schedule_interval='0 18 * * *',  # 매일 18:00(KST)에 실행
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