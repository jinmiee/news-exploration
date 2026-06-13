# 뉴스대탐험 - Django 웹 서비스 이미지
FROM python:3.12-slim

# konlpy(Okt) 형태소 분석기는 JVM이 필요하므로 JDK 설치.
# build-essential 은 일부 패키지(C 확장) 빌드용.
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jdk \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/default-java

WORKDIR /app

# 의존성 레이어 캐싱: requirements 가 바뀌지 않으면 재설치하지 않음
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 복사
COPY . .

EXPOSE 8000

# 프로덕션 WSGI 서버(gunicorn). 개발 시에는 docker-compose 에서 runserver 로 override.
CMD ["gunicorn", "news.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120"]
