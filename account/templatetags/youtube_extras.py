# account/templatetags/youtube_extras.py
from django import template
import re

register = template.Library()

@register.filter
def youtube_id(url):
    # 정규식을 사용하여 유튜브 ID 추출
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else ''
