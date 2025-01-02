from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from django.core.mail import send_mail
from pymongo import MongoClient


def send_email_task():
    print("send_email_task 실행됨")  # 디버깅 로그
    from django.contrib.auth.models import User  # 함수 내부에서 import하여 초기화 시점 문제 방지
    users = User.objects.all()
    subject = "그게 뭔데?! 차트업데이트 알림"
    message = "새로운 뉴스가 업데이트 되었어요! 확인해보세요!. \n 뉴스 보러가기 --> http://15.168.46.71:5070/account/chart"
    from_email = "namsugb99@gmail.com"
    recipient_list = [user.email for user in users if user.email]

    send_mail(subject, message, from_email, recipient_list)

def start_scheduler(save_chart_to_mongo=None):
    from .processing_tasks import save_daily_top10, delete_expired_charts, save_top10_to_chart, extract_duplicates_for_weekly_issues, extract_duplicates_for_chart
    import logging

    logging.basicConfig()
    logging.getLogger('apscheduler').setLevel(logging.DEBUG)

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

    # 스케줄러 상태 확인 함수
    def ensure_scheduler_running():
        if not scheduler.running or scheduler.state == 0:  # 0은 종료 상태
            scheduler.start()
            print("스케줄러가 시작되었습니다.")

    # 중복 등록 방지 및 기존 작업 제거
    def add_job_if_not_exists(job_id, func, trigger, **kwargs):
        if not scheduler.get_job(job_id):
            scheduler.add_job(func, trigger=trigger, id=job_id, replace_existing=True, **kwargs)

    # 스케줄러 시작
    ensure_scheduler_running()

    # 작업 추가
    add_job_if_not_exists(
        'send_email',
        send_email_task,
        trigger=CronTrigger(hour='11,23', minute=5),
        max_instances=3  # 중복 실행 허용 추가
    )

    add_job_if_not_exists(
        'save_daily_top10',
        save_daily_top10,
        trigger=CronTrigger(hour=0, minute=5),
        max_instances = 3  # 최대 3개까지 중복 실행 허용
    )
    print("daily_top10 저장 작업이 실행됩니다.")

    add_job_if_not_exists(
        'save_top10_to_chart',
        save_top10_to_chart,
        trigger=CronTrigger(hour='*', minute=5),  # 매시 5분에 실행
        max_instances=3  # 최대 3개까지 중복 실행 허용
    )
    print("Chart 저장 작업이 매시 5분에 실행되도록 등록되었습니다.")

    add_job_if_not_exists(
        'extract_weekly_duplicates',
        extract_duplicates_for_weekly_issues,
        trigger=CronTrigger(hour=0, minute=8),
        max_instances=3  # 중복 실행 허용 추가
    )
    print("Weekly Issues 중복 데이터 추출 작업이 실행됩니다.")

    add_job_if_not_exists(
        'extract_chart_duplicates',
        extract_duplicates_for_chart,
        trigger=CronTrigger(hour='*', minute=8),  # 매시 5분에 실행
        max_instances=3  # 중복 실행 허용 추가
    )
    print("Chart 중복 데이터 추출 작업이 매시 5분에 실행되도록 등록되었습니다.")

    add_job_if_not_exists(
        'delete_expired_charts',
        delete_expired_charts,
        trigger=CronTrigger(hour=1)
    )
    print("스케줄러가 시작되었습니다: 24시간 지난 Chart 데이터를 삭제합니다.")

    # Django 종료 시 스케줄러 중지
    def stop_scheduler():
        if scheduler.running:
            scheduler.shutdown(wait=True)  # 실행 중인 작업 완료 후 종료

    import atexit
    atexit.register(stop_scheduler)