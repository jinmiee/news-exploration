from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from django.core.mail import send_mail
from pymongo import MongoClient
from .views import save_daily_top10

def send_email_task():
    from django.contrib.auth.models import User  # 함수 내부에서 import하여 초기화 시점 문제 방지
    users = User.objects.all()
    subject = "차트업데이트 알림"
    message = "새로운 뉴스가 업데이트 되었어요! 확인해보세요!."
    from_email = "namsugb99@gmail.com"
    recipient_list = [user.email for user in users if user.email]

    send_mail(subject, message, from_email, recipient_list)


def start_scheduler():
    jobstores = {
        'default': MongoDBJobStore(
            database='youtube_data',
            collection='apscheduler_jobs',
            client=MongoClient(
                host='mongodb://hello-news.site:27777',
                username='entks',
                password='entks',
                authSource='admin',
                authMechanism='SCRAM-SHA-256'
            )
        )
    }

    executors = {'default': ThreadPoolExecutor(20)}
    scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, timezone="Asia/Seoul")

    # 작업 추가: 이메일 알림
    scheduler.add_job(
        send_email_task,
        trigger=CronTrigger(hour='11,23', minute=5),
        id='send_email',
        replace_existing=True
    )

    # 작업 추가: daily_top10 저장
    scheduler.add_job(
        save_daily_top10,
        trigger=CronTrigger(hour=0, minute=0),
        id='daily_top10',
        replace_existing=True
    )

    print("스케줄러가 시작되었습니다.")
    scheduler.start()

    # Django 서버 종료 시 스케줄러 중지
    import atexit
    atexit.register(lambda: scheduler.shutdown())
