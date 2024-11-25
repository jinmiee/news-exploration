from djongo import models

class YouTubeComment(models.Model):
    author = models.CharField(max_length=255, null=True, blank=True)  # 댓글 작성자
    comment = models.TextField(null=True, blank=True)  # 댓글 내용

    def __init__(self, *args, **kwargs):
        try:
            if args and isinstance(args[0], str):
                kwargs['author'] = "Unknown"
                kwargs['comment'] = args[0]
        except Exception as e:
            print(f"Error initializing YouTubeComment: {e}")
        super().__init__(*args, **kwargs)

    class Meta:
        abstract = True


class YouTubeData(models.Model):
    id = models.ObjectIdField(primary_key=True)  # MongoDB ObjectId
    channel_name = models.CharField(max_length=255)  # 채널 이름
    title = models.TextField()  # 동영상 제목
    views = models.BigIntegerField()  # 조회수 (큰 숫자 처리)
    upload_date = models.DateTimeField()  # 업로드 날짜
    url = models.URLField()  # 동영상 URL
    channel = models.CharField(max_length=255)  # 채널 제목
    desc = models.TextField(blank=True, null=True)  # 동영상 설명 (옵션)
    likes = models.BigIntegerField(blank=True, null=True)  # 좋아요 수
    comments = models.ArrayField(
        model_container=YouTubeComment,  # 댓글 배열
        blank=True,
        null=True
    )

    class Meta:
        db_table = "youtube_datas"  # MongoDB 컬렉션 이름