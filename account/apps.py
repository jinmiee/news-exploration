from django.apps import AppConfig
import threading

class AccountConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'account'

    def ready(self):
        from .tasks import start_scheduler

        # 스케줄러를 별도의 스레드에서 실행
        threading.Thread(target=start_scheduler, daemon=True).start()
