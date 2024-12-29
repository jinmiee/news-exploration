from django.apps import AppConfig

class AccountConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'account'

    def ready(self):
        from django.conf import settings

        # 스케줄러 활성화 여부를 settings에서 확인
        if getattr(settings, 'ENABLE_SCHEDULER', True):
            try:
                # 스케줄러 Import를 지연하여 순환 참조 방지
                from account.tasks.scheduler import start_scheduler

                # 스케줄러가 이미 실행 중인지 확인
                if not hasattr(self, 'scheduler_started'):
                    print("스케줄러를 시작합니다.")
                    start_scheduler()
                    self.scheduler_started = True  # 중복 실행 방지 플래그 설정
            except ImportError as e:
                print(f"스케줄러를 시작할 수 없습니다: {e}")
            except Exception as e:
                print(f"스케줄러 실행 중 오류 발생: {e}")
