from django.apps import AppConfig
import threading

class AccountConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'account'

    def ready(self):
        # 스케줄러를 지연 실행
        from django.conf import settings
        if settings.DEBUG:  # 개발 모드에서만 스케줄러 실행
            import threading
            from .tasks import start_scheduler
            threading.Thread(target=start_scheduler).start()