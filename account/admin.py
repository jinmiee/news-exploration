from django.contrib import admin

# Register your models here.

from .models import YouTubeData, YouTubeComment, Like

@admin.register(Like)
class LikeAdmin(admin.ModelAdmin):
    list_display = ('user', 'youtube_data')
    list_filter = ('user', 'youtube_data')
    search_fields = ('user', 'youtube_data')
    