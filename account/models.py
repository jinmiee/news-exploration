from djongo import models
from django.contrib.auth.models import User


class YouTubeComment(models.Model): # 유튜브 댓글 모델
    author = models.CharField(max_length=255, null=True, blank=True)  # 댓글 작성자
    comment = models.TextField(null=True, blank=True)  # 댓글 내용

    def __init__(self, *args, **kwargs): #초기화 메서드
        try:
            if args and isinstance(args[0], str): #첫번째 인자가 문자열일 경우
                kwargs['author'] = "Unknown" # 작성자를 "Unknown" 으로 설정
                kwargs['comment'] = args[0] # 댓글 내용을 해당 문자열로 설정
        except Exception as e:
            print(f"Error initializing YouTubeComment: {e}") #오류 메시지 출력
        super().__init__(*args, **kwargs)

    class Meta:
        abstract = True # 이 모델은 추상 모델로 데이터베이스에 직접 생성되지 않음.

class Transcript(models.Model):  # 스크립트 모델
    start = models.FloatField()  # 스크립트 시작 시간
    text = models.TextField()  # 스크립트 텍스트

    class Meta:
        abstract = True  # 추상 모델로 설정

class YouTubeData(models.Model):
    _id = models.ObjectIdField(primary_key=True)  # MongoDB ObjectId를 기본 키로 설정
    channel_name = models.CharField(max_length=255)  # 채널 이름
    title = models.TextField()  # 동영상 제목
    views = models.BigIntegerField()  # 조회수 (큰 숫자 처리)
    upload_date = models.DateTimeField()  # 업로드 날짜
    url = models.URLField(unique=True)  # 동영상 URL
    channel = models.CharField(max_length=255)  # 채널 제목
    desc = models.TextField(blank=True, null=True)  # 동영상 설명 (옵션)
    likes = models.BigIntegerField(blank=True, null=True)  # 좋아요 수
    thumbnail = models.URLField(blank=True, null=True)  # 썸네일 URL 추가
    comments = models.ArrayField(
        model_container=YouTubeComment,  # 댓글 모델 지정
        blank=True,
        null=True
    )
    transcript = models.ArrayField(
        model_container=Transcript,
        blank=True,
        null=True,
        default=list  # 기본값으로 빈 리스트를 설정
    )

    class Meta:
        db_table = "youtube_datas"  # MongoDB 컬렉션 이름


class Like(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    youtube_data = models.ForeignKey(YouTubeData, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('user', 'youtube_data')

    def __str__(self):
        return f"{self.user.username} likes {self.youtube_data.title}"