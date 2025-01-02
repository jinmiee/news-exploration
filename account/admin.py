from django.contrib import admin

# Register your models here.

from .models import YouTubeData, YouTubeComment, Like, Feedbacks

@admin.register(Like)
class LikeAdmin(admin.ModelAdmin):
    list_display = ('user', 'youtube_data')
    list_filter = ('user', 'youtube_data')
    search_fields = ('user', 'youtube_data')


@admin.register(Feedbacks)
class FeedbacksAdmin(admin.ModelAdmin):
    list_display = ('user', 'feedback', 'rating', 'created_at')
    list_filter = ('user', 'feedback')
    search_fields = ('user', 'feedback')

    def has_delete_permission(self, request, obj=None):
        # 삭제 권한 확인 (True로 설정되어야 삭제 가능)
        return True