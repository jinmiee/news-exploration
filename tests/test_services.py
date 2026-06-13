"""
account.services 의 차트 시간대 계산 로직 단위 테스트.
(뷰에서 분리한 순수 함수 — Django/DB 없이 실행 가능)
"""
from datetime import datetime, timedelta

import pytz

from account.services import get_chart_query_window, get_chart_display_window

KST = pytz.timezone("Asia/Seoul")


def _at(hour, minute=0):
    """2026-06-14 의 특정 시각(KST)을 반환."""
    return KST.localize(datetime(2026, 6, 14, hour, minute, 0))


class TestChartQueryWindow:
    def test_오전11시_이전이면_전날_11시부터_23시까지(self):
        now = _at(9)
        start, end = get_chart_query_window(now)
        assert (start.hour, start.minute) == (11, 0)
        assert (end.hour, end.minute) == (23, 0)
        assert start.date() == (now - timedelta(days=1)).date()
        assert end.date() == (now - timedelta(days=1)).date()

    def test_낮시간이면_전날23시부터_오늘11시까지(self):
        now = _at(15)
        start, end = get_chart_query_window(now)
        assert (start.hour, start.date()) == (23, (now - timedelta(days=1)).date())
        assert (end.hour, end.date()) == (11, now.date())

    def test_오후11시_이후면_오늘11시부터_23시까지(self):
        now = _at(23)
        start, end = get_chart_query_window(now)
        assert (start.hour, start.date()) == (11, now.date())
        assert (end.hour, end.date()) == (23, now.date())


class TestChartDisplayWindow:
    def test_갱신11시7분_이전(self):
        now = _at(9)
        update, start, end = get_chart_display_window(now)
        assert (update.hour, update.minute, update.date()) == (11, 7, now.date())
        assert (start.hour, start.minute, start.date()) == (23, 7, (now - timedelta(days=1)).date())
        assert (end.hour, end.minute, end.date()) == (11, 7, now.date())

    def test_갱신11시7분과_23시7분_사이(self):
        now = _at(15)
        update, start, end = get_chart_display_window(now)
        assert (update.hour, update.minute, update.date()) == (23, 7, now.date())
        assert (start.hour, start.minute, start.date()) == (11, 7, now.date())
        assert (end.hour, end.minute, end.date()) == (23, 7, now.date())

    def test_갱신23시7분_이후(self):
        now = _at(23, 30)
        update, start, end = get_chart_display_window(now)
        assert (update.hour, update.minute, update.date()) == (11, 7, (now + timedelta(days=1)).date())
        assert (start.hour, start.minute, start.date()) == (23, 7, now.date())
        assert (end.hour, end.minute, end.date()) == (11, 7, (now + timedelta(days=1)).date())
