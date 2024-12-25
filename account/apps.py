from django.apps import AppConfig
import threading
import time

class AccountConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'account'

    def ready(self):
        # 스케줄러 실행을 설정에 따라 제어
        from django.conf import settings
        if getattr(settings, 'ENABLE_SCHEDULER', True):
            def delayed_scheduler():
                time.sleep(1)  # 1초 지연
                from account.tasks.scheduler import start_scheduler  # Import를 지연시켜 순환 참조 방지
                start_scheduler()

            threading.Thread(target=delayed_scheduler, daemon=True).start()