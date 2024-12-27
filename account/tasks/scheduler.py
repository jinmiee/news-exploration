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
    subject = "차트업데이트 알림"
    message = "새로운 뉴스가 업데이트 되었어요! 확인해보세요!."
    from_email = "namsugb99@gmail.com"
    recipient_list = [user.email for user in users if user.email]

    send_mail(subject, message, from_email, recipient_list)

def start_scheduler(save_chart_to_mongo=None):
    from .processing_tasks import save_daily_top10, delete_expired_charts, save_top10_to_chart, extract_duplicates_for_weekly_issues, extract_duplicates_for_chart
    from .related_word_tasks import save_related_word_analysis

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

    # 중복 등록 방지 로직
    def add_job_if_not_exists(job_id, func, trigger, **kwargs):
        if not scheduler.get_job(job_id):
            scheduler.add_job(func, trigger=trigger, id=job_id, replace_existing=True, **kwargs)

    # 작업 추가
    add_job_if_not_exists(
        'send_email',
        send_email_task,
        trigger=CronTrigger(hour='11,23', minute=5)
    )

    add_job_if_not_exists(
        'save_daily_top10',
        save_daily_top10,
        trigger=CronTrigger(hour=0, minute=5)
    )
    print("daily_top10 저장 작업이 실행됩니다.")

    add_job_if_not_exists(
        'save_top10_to_chart',
        save_top10_to_chart,
        trigger=IntervalTrigger(hours=1)
    )
    print("Chart 저장 작업이 1시간마다 실행되도록 등록되었습니다.")

    add_job_if_not_exists(
        'extract_weekly_duplicates',
        extract_duplicates_for_weekly_issues,
        trigger=CronTrigger(hour=0, minute=5)
    )
    print("Weekly Issues 중복 데이터 추출 작업이 실행됩니다.")

    add_job_if_not_exists(
        'extract_chart_duplicates',
        extract_duplicates_for_chart,
        trigger=IntervalTrigger(hours=5)
    )
    print("Chart 중복 데이터 추출 작업이 1시간마다 실행되도록 등록되었습니다.")

    add_job_if_not_exists(
        'delete_expired_charts',
        delete_expired_charts,
        trigger=CronTrigger(hour=0, minute=5)
    )
    print("스케줄러가 시작되었습니다: 24시간 지난 Chart 데이터를 삭제합니다.")

    add_job_if_not_exists(
        'save_related_word_analysis',
        save_related_word_analysis,
        trigger=CronTrigger(hour='11,23', minute=5)
    )
    print("연관어 분석 작업이 매일 11시 5분, 23시 5분에 실행되도록 등록되었습니다.")

    # 스케줄러 시작
    scheduler.start()
    print("스케쥴링이 실행되었습니다")
    scheduler.print_jobs()

    # Django 종료 시 스케줄러 중지
    import atexit
    atexit.register(lambda: scheduler.shutdown())

    try:
        # MongoDB 연결 테스트
        client = MongoClient(
            host='mongodb://hello-news.site:27777',
            username='entks',
            password='entks',
            authSource='admin',
            authMechanism='SCRAM-SHA-256'
        )
        db = client['youtube_data']
        print("MongoDB 연결 성공")
    except Exception as e:
        print(f"MongoDB 연결 실패: {str(e)}")
        return

def check_scheduler_status():
    """스케줄러 상태 확인"""
    try:
        client = MongoClient(
            host='mongodb://hello-news.site:27777',
            username='entks',
            password='entks',
            authSource='admin',
            authMechanism='SCRAM-SHA-256'
        )
        db = client['youtube_data']
        jobs = db['apscheduler_jobs'].find()
        
        print("\n=== 스케줄러 작업 목록 ===")
        for job in jobs:
            print(f"작업 ID: {job['_id']}")
            print(f"다음 실행 시간: {job.get('next_run_time')}")
            print("------------------------")
            
    except Exception as e:
        print(f"스케줄러 상태 확인 중 오류 발생: {str(e)}")