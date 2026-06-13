"""
뷰에서 분리한 비즈니스 로직(서비스 레이어).

순수 함수로 작성되어 Django/DB 의존성이 없으므로 단위 테스트가 용이하다.
차트 시간대 계산처럼 뷰 안에 뒤엉켜 있던 로직을 이곳으로 옮겨
뷰는 "요청 처리"에만 집중하도록 한다.
"""
from datetime import timedelta


def get_chart_query_window(now):
    """
    차트 DB 조회에 사용할 분석 구간을 계산한다 (분=00 기준).

    - 오전 11시 이전: 전날 11시 ~ 전날 23시
    - 11시~23시 사이: 전날 23시 ~ 오늘 11시
    - 23시 이후: 오늘 11시 ~ 오늘 23시

    Returns:
        (analysis_start, analysis_end): tz-aware datetime 튜플
    """
    if now.hour < 11:
        start = (now - timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)
        end = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
    elif now.hour < 23:
        start = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
        end = now.replace(hour=11, minute=0, second=0, microsecond=0)
    else:
        start = now.replace(hour=11, minute=0, second=0, microsecond=0)
        end = now.replace(hour=23, minute=0, second=0, microsecond=0)
    return start, end


def get_chart_display_window(now):
    """
    화면에 표시할 다음 갱신 시각과 표시 구간을 계산한다 (분=07 기준).
    차트는 11:07 / 23:07 에 갱신되므로, 현재 시각에 따라 다음 갱신 시각과
    그에 대응하는 분석 구간을 돌려준다.

    Returns:
        (chart_update_time, display_start, display_end)
    """
    if now < now.replace(hour=11, minute=7, second=0, microsecond=0):
        update = now.replace(hour=11, minute=7, second=0, microsecond=0)
        start = (now - timedelta(days=1)).replace(hour=23, minute=7, second=0, microsecond=0)
        end = now.replace(hour=11, minute=7, second=0, microsecond=0)
    elif now < now.replace(hour=23, minute=7, second=0, microsecond=0):
        update = now.replace(hour=23, minute=7, second=0, microsecond=0)
        start = now.replace(hour=11, minute=7, second=0, microsecond=0)
        end = now.replace(hour=23, minute=7, second=0, microsecond=0)
    else:
        update = (now + timedelta(days=1)).replace(hour=11, minute=7, second=0, microsecond=0)
        start = now.replace(hour=23, minute=7, second=0, microsecond=0)
        end = (now + timedelta(days=1)).replace(hour=11, minute=7, second=0, microsecond=0)
    return update, start, end
