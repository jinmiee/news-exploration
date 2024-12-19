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

