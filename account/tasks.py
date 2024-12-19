<<<<<<< HEAD
from apscheduler.schedulers.background import BackgroundScheduler
from django.core.mail import send_mail

def send_email_task():
    from django.contrib.auth.models import User  # 함수 내부에서 import하여 초기화 시점 문제 방지
    users = User.objects.all()
    subject = "차트업데이트 알림"
    message = "새로운 뉴스가 업데이트 되었어요! 확인해보세요!."
    from_email = "namsugb99@gmail.com"
    recipient_list = [user.email for user in users if user.email]

    send_mail(subject, message, from_email, recipient_list)



def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_email_task, 'cron', hour='11,23', minute='0')
    print("스케줄러가 시작되었습니다.")  # 스케줄러 시작 로그
    print("현재 작업 목록:", scheduler.get_jobs())  # 등록된 작업 로그
    scheduler.start()

=======
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.executors.pool import ThreadPoolExecutor
from pymongo import MongoClient

def start_scheduler():
    from .views import save_daily_top10  # 함수 호출 시점에 임포트
    jobstores = {
        'default': MongoDBJobStore(
            database='youtube_data',  # 사용할 데이터베이스
            collection='apscheduler_jobs',  # 스케줄러 작업이 저장될 컬렉션
            client=MongoClient(
                host='mongodb://hello-news.site:27777',  # MongoDB 서버 주소
                username='entks',  # 사용자 이름
                password='entks',  # 비밀번호
                authSource='admin',  # 인증 소스
                authMechanism='SCRAM-SHA-256'  # SCRAM-SHA-256 명시
            )
        )
    }

    executors = {'default': ThreadPoolExecutor(20)}
    scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors)

    # 작업 추가
    scheduler.add_job(
        save_daily_top10,
        trigger=CronTrigger(hour=0, minute=0),
        id='daily_top10',
        replace_existing=True
    )
    print("Scheduler started")
    scheduler.start()
>>>>>>> 2ea2cd2057632b5fe21d944fd30300cf38f46e42
