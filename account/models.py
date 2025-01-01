from bson import ObjectId
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
        model_container=YouTubeComment,
        blank=True,
        null=False,  # null 값 허용하지 않음
        default=list  # 기본값을 빈 리스트로 설정
    )
    transcript = models.ArrayField(
        model_container=Transcript,
        blank=True,
        null=False,  # null 값을 허용하지 않음
        default=list
    )

    class Meta:
        db_table = "youtube_datas"  # MongoDB 컬렉션 이름


class Like(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    youtube_data = models.ForeignKey(YouTubeData, on_delete=models.CASCADE)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'youtube_data'], name='unique_user_like')
        ]

    def __str__(self):
        return f"{self.user.username} likes {self.youtube_data.title}"

class WeeklyIssue(models.Model):
    _id = models.ObjectIdField(primary_key=True, default=ObjectId)
    channel_name = models.CharField(max_length=255)
    title = models.TextField()
    views = models.BigIntegerField()
    upload_date = models.DateTimeField()
    url = models.URLField(unique=True)
    channel = models.CharField(max_length=255)
    thumbnail = models.URLField(blank=True, null=True)
    comments = models.ArrayField(
        model_container=YouTubeComment,
        blank=True,
        null=False,
        default=list
    )
    transcript = models.ArrayField(
        model_container=Transcript,
        blank=True,
        null=False,
        default=list
    )
    desc = models.TextField(blank=True, null=True)  # Add this field
    likes = models.BigIntegerField(blank=True, null=True)  # Add this field
    rank = models.IntegerField(blank=True, null=True)  # Add this field

    class Meta:
        db_table = "weekly_issues"
        indexes = [
            models.Index(fields=['title']),  # title 필드에 인덱스 추가
            models.Index(fields=['url']),    # url 필드에 인덱스 추가
        ]


class Chart(models.Model):
    _id = models.ObjectIdField(primary_key=True, default=ObjectId)
    rank = models.IntegerField()
    channel_name = models.CharField(max_length=255)
    title = models.TextField()
    views = models.BigIntegerField()
    upload_date = models.DateTimeField()
    url = models.URLField(unique=True)
    channel = models.CharField(max_length=255)
    desc = models.TextField(blank=True, null=True)
    likes = models.BigIntegerField(blank=True, null=True)
    thumbnail = models.URLField(blank=True, null=True)
    comments = models.ArrayField(
        model_container=YouTubeComment,
        blank=True,
        null=False,
        default=list
    )
    transcript = models.ArrayField(
        model_container=Transcript,
        blank=True,
        null=False,
        default=list
    )

    class Meta:
        db_table = "chart"
        indexes = [
            models.Index(fields=['title']),  # 텍스트 인덱스는 유지
        ]


class Feedbacks(models.Model):
    _id = models.ObjectIdField(primary_key=True, default=ObjectId)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.PositiveBigIntegerField()
    feedback = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "feedbacks"  # MongoDB 컬렉션 이름


class TranscriptItem(models.Model):  # ArrayField에 사용할 모델
    start = models.FloatField()  # 시작 시간
    text = models.TextField()  # 텍스트 내용

    class Meta:
        abstract = True  # 데이터베이스에 직접 테이블을 생성하지 않음

class ChartDuplicateVideo(models.Model):
    _id = models.ObjectIdField(primary_key=True, default=ObjectId)
    title = models.TextField()
    url = models.URLField(unique=True)
    views = models.BigIntegerField()
    upload_date = models.DateTimeField()
    channel_name = models.CharField(max_length=255)
    thumbnail = models.URLField(blank=True, null=True)
    likes = models.BigIntegerField(blank=True, null=True)
    transcript = models.ArrayField(
        model_container=TranscriptItem,  # 수정된 부분
        blank=True,
        null=False,
        default=list
    )

    class Meta:
        db_table = "chart_duplicate_videos"
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['upload_date']),
            models.Index(fields=['views']),
        ]

class WeeklyIssueDuplicateVideo(models.Model):
    _id = models.ObjectIdField(primary_key=True, default=ObjectId)
    title = models.TextField()
    url = models.URLField(unique=True)
    views = models.BigIntegerField()
    upload_date = models.DateTimeField()
    channel_name = models.CharField(max_length=255)
    thumbnail = models.URLField(blank=True, null=True)
    likes = models.BigIntegerField(blank=True, null=True)
    transcript = models.ArrayField(
        model_container=TranscriptItem,  # 수정된 부분
        blank=True,
        null=False,
        default=list
    )

    class Meta:
        db_table = "weekly_issue_duplicate_videos"
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['upload_date']),
            models.Index(fields=['channel_name']),
        ]

class RelatedWordAnalysis(models.Model):
    video_id = models.CharField(max_length=255)  # 비디오 URL 또는 ID
    graph_image = models.TextField()  # Base64로 인코딩된 네트워크 그래프 이미지
    top_pairs = models.JSONField()  # 상위 연관어 쌍 데이터
    keywords = models.JSONField()  # 키워드 리스트
    title = models.TextField()  # 동영상 제목
    _id = models.ObjectIdField(primary_key=True, default=ObjectId)
    important_keywords = models.JSONField()  # 중요 키워드 리스트
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'chart_relation_analysis'