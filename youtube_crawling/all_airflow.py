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
                    comments.append(comment)
                    
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
            print(f"댓글 수집 오류: {e}")  # 오류 발생 시 출력
        return comments

    def collect_channel_data(self, channel_name, channel_id):
        """
        특정 채널의 최근 24시간 동안의 YouTube 데이터를 수집하는 메서드.

        Args:
            channel_name (str): 채널 이름.
            channel_id (str): YouTube 채널 ID.
        """
        # KST(한국 표준시) 기준으로 현재 시간과 24시간 전 시간 계산
        now_kst = datetime.now(self.KST)
        yesterday_kst = now_kst - timedelta(hours=24)
        yesterday_utc = yesterday_kst.astimezone(pytz.utc)  # UTC로 변환
        start_time = yesterday_utc.strftime('%Y-%m-%dT%H:%M:%SZ')  # ISO 형식으로 변환

        videos = []  # 수집된 동영상 데이터를 저장할 리스트
        page_token = None

        try:
            while True:
                # YouTube API를 통해 동영상 검색
                search_response = self.youtube.search().list(
                    order='date',  # 최신순으로 정렬
                    part='snippet',
                    channelId=channel_id,
                    maxResults=50,  # 한 번에 최대 50개 검색
                    publishedAfter=start_time,  # 24시간 이내의 데이터만 가져옴
                    type='video',
                    pageToken=page_token  # 페이지 토큰으로 다음 페이지 검색
                ).execute()

                # 동영상 ID 리스트 추출
                video_ids = [item['id']['videoId'] for item in search_response['items']]

                # 동영상 세부 정보 요청
                videos_response = self.youtube.videos().list(
                    part='snippet,statistics',  # 메타데이터와 통계 포함
                    id=','.join(video_ids)
                ).execute()

                for video in videos_response['items']:
                    # 실시간 방송은 제외
                    if video['snippet'].get('liveBroadcastContent') == 'none':
                        published_at = video['snippet']['publishedAt']
                        published_at_kst = datetime.strptime(
                            published_at, '%Y-%m-%dT%H:%M:%SZ'
                        ).replace(tzinfo=timezone.utc).astimezone(self.KST)  # UTC -> KST 변환

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

                # 다음 페이지가 있으면 계속 검색, 최대 100개까지만 수집
                if 'nextPageToken' in search_response and len(videos) < 100:
                    page_token = search_response['nextPageToken']
                else:
                    break

            # 조회수 기준으로 정렬하여 상위 20개만 저장
            videos.sort(key=lambda x: int(x['views']), reverse=True)
            videos = videos[:20]

            if videos:
                mongo_client = self.mongo_hook.get_conn()  # MongoDB 연결
                db = mongo_client['youtube_data']  # 데이터베이스 선택
                collection = db['youtube_datas']  # 컬렉션 선택
                collection.insert_many(videos)  # 데이터 저장
                print(f"{channel_name} 데이터 저장 완료 (상위 20개)")

        except Exception as e:
            print(f"{channel_name} 데이터 수집 오류: {e}")  # 오류 발생 시 출력

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
        catchup=False  # 이전 날짜의 DAG 실행 방지
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
        python_callable=lambda: run_youtube_data_collection('Your_YouTube_API_Key'),  # API 키 전달
        dag=dag
    )
    collect_youtube_data