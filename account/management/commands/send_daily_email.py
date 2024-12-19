from django.core.management.base import BaseCommand
from account.utils import send_daily_email

class Command(BaseCommand):
    help = 'Send daily email to users with notifications enabled'

    def handle(self, *args, **kwargs):
        send_daily_email()
        self.stdout.write(self.style.SUCCESS('Daily email sent successfully!'))
