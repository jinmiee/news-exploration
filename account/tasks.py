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