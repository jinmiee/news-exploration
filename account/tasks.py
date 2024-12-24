from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from django.core.mail import send_mail
from pymongo import MongoClient

def send_email_task():
    from django.contrib.auth.models import User  # 함수 내부에서 import하여 초기화 시점 문제 방지
    users = User.objects.all()
    subject = "차트업데이트 알림"
    message = "새로운 뉴스가 업데이트 되었어요! 확인해보세요!."
    from_email = "namsugb99@gmail.com"
    recipient_list = [user.email for user in users if user.email]

    send_mail(subject, message, from_email, recipient_list)


def start_scheduler(save_chart_to_mongo=None):
    from .views import save_daily_top10, delete_expired_charts, save_top10_to_chart,extract_duplicates_for_weekly_issues, extract_duplicates_for_chart
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

    # 상위 10개 저장: daily_top10
    scheduler.add_job(
        save_daily_top10,
        trigger=IntervalTrigger(hours=1),
        id='save_daily_top10',
        replace_existing=True
    )
    print("daily_top10 저장 작업이 1시간마다 실행되도록 등록되었습니다.")

    # 상위 10개 저장: chart
    scheduler.add_job(
        save_top10_to_chart,
        trigger=IntervalTrigger(hours=1),
        id='save_top10_to_chart',
        replace_existing=True
    )
    print("Chart 저장 작업이 1시간마다 실행되도록 등록되었습니다.")

    # 중복 데이터 추출: weekly issues
    scheduler.add_job(
        extract_duplicates_for_weekly_issues,
        trigger=IntervalTrigger(hours=1),
        id='extract_weekly_duplicates',
        replace_existing=True
    )
    print("Weekly Issues 중복 데이터 추출 작업이 1시간마다 실행되도록 등록되었습니다.")

    # 중복 데이터 추출: chart
    scheduler.add_job(
        extract_duplicates_for_chart,
        trigger=IntervalTrigger(hours=1),
        id='extract_chart_duplicates',
        replace_existing=True
    )
    print("Chart 중복 데이터 추출 작업이 1시간마다 실행되도록 등록되었습니다.")

    # 24시간 지난 Chart 데이터를 삭제하는 작업 추가
    scheduler.add_job(
        delete_expired_charts,
        trigger=IntervalTrigger(hours=1),  # 매 1시간마다 실행
        id="delete_expired_charts",
        replace_existing=True
    )
    print("스케줄러가 시작되었습니다: 24시간 지난 Chart 데이터를 삭제합니다.")

    # 스케줄러 시작
    scheduler.start()
    print("스케줄러가 시작되었습니다.")

    # Django 종료 시 스케줄러 중지
    import atexit
    atexit.register(lambda: scheduler.shutdown())
