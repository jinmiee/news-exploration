from django.core.management.base import BaseCommand
from account.tasks.related_word_tasks import save_related_word_analysis
from account.models import Chart

class Command(BaseCommand):
    help = 'Test related words analysis for top 10 chart videos'

    def handle(self, *args, **options):
        self.stdout.write('연관어 분석 테스트 시작...')
        try:
            # 차트의 상위 10개 뉴스에 대한 분석 실행
            save_related_word_analysis()
            self.stdout.write(self.style.SUCCESS('연관어 분석 완료'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'오류 발생: {str(e)}')) 