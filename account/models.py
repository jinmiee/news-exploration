from djongo import models

class YouTubeData(models.Model):
    id = models.ObjectIdField()
    channel = models.CharField(max_length=255)
    channel_name = models.CharField(max_length=255)
    # comments = models.ArrayField(
    #     model_container=YouTubeComment,
    #     null=True,
    #     blank=True
    # )
    desc = models.TextField()
    likes = models.IntegerField()
    title = models.CharField(max_length=500)
    upload_date = models.DateTimeField()
    url = models.URLField()
    views = models.IntegerField()

    class Meta:
        db_table = 'youtube_datas'