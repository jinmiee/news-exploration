from django.core.management.base import BaseCommand
from account.tasks.scheduler import check_scheduler_status

class Command(BaseCommand):
    help = 'Check scheduler status'

    def handle(self, *args, **options):
        check_scheduler_status() 