from django.db import models

class Article(models.Model):
    # MongoDB의 'articles' 컬렉션과 일치하도록 모델 필드 정의
    title = models.CharField(max_length=255)
    content = models.TextField()

    class Meta:
        # MongoDB의 기존 컬렉션을 사용하도록 설정
        db_table = 'youtube_data'