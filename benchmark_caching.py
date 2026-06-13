"""
DB 캐싱 효과 측정 벤치마크.

핵심 아이디어:
  - 뉴스 선정/중복제거/연관어 분석의 비용은 대부분 KoBERT 임베딩 계산이다.
  - 이 프로젝트는 스케줄러가 분석을 *사전 계산*해 MongoDB에 저장하고,
    사용자 요청 시에는 결과를 *조회만* 한다.
  - 따라서 "매 요청 분석" vs "사전계산 후 DB 조회"의 응답시간 차이를 측정한다.

실행: ./venv/Scripts/python.exe benchmark_caching.py
"""
import os
import time
import logging

logging.disable(logging.CRITICAL)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "news.settings")

import django
django.setup()

from account.models import YouTubeData
from account.analysis.text_processing import process_text, get_bert_similarity_batch

N = 15  # 비교 대상 기사 수 (차트 Top 선정 규모와 유사)


def fmt(sec):
    return f"{sec*1000:.1f} ms" if sec < 1 else f"{sec:.2f} s"


def main():
    # 1) 자막 보유 기사 N개 로드
    videos = [v for v in YouTubeData.objects.exclude(transcript=[])[: N * 3]][:N]
    print(f"대상 기사 수: {len(videos)}")

    # 2) [캐시 읽기] 사전계산 결과를 조회만 하는 경로 ─ 인덱스 단건 조회 5회 중앙값
    sample_url = videos[0].url
    reads = []
    for _ in range(5):
        t = time.perf_counter()
        _ = YouTubeData.objects.filter(url=sample_url).first()
        reads.append(time.perf_counter() - t)
    reads.sort()
    t_cached = reads[len(reads) // 2]

    # 3) [매 요청 분석] 전처리 + KoBERT 임베딩 유사도 계산
    t = time.perf_counter()
    corpus = [process_text(v.title, v.transcript or []) for v in videos]
    t_pre = time.perf_counter() - t

    # 모델 로드(최초 1회 비용) 분리 측정
    t = time.perf_counter()
    _ = get_bert_similarity_batch(corpus[:2])  # 워밍업 겸 모델 로드
    t_warm = time.perf_counter() - t

    t = time.perf_counter()
    _ = get_bert_similarity_batch(corpus)
    t_infer = time.perf_counter() - t

    t_compute_cold = t_pre + t_warm
    t_compute_warm = t_pre + t_infer

    # 4) 결과 출력
    print("\n================ 결과 ================")
    print(f"[사전계산+캐시 조회]  단건 DB 조회        : {fmt(t_cached)}")
    print(f"[매 요청 분석]        전처리({len(videos)}건)        : {fmt(t_pre)}")
    print(f"[매 요청 분석]        KoBERT 추론({len(videos)}건)   : {fmt(t_infer)}")
    print(f"[매 요청 분석]        모델 로드(콜드 1회)  : {fmt(t_warm)}")
    print("--------------------------------------")
    print(f"분석 1회(웜)  총      : {fmt(t_compute_warm)}")
    print(f"분석 1회(콜드)총      : {fmt(t_compute_cold)}")
    print(f"캐시 조회             : {fmt(t_cached)}")
    if t_cached > 0:
        print(f"\n→ 캐싱 효과: 약 {t_compute_warm / t_cached:.0f}배 빠름 (웜 기준), "
              f"{t_compute_cold / t_cached:.0f}배 (콜드 기준)")


if __name__ == "__main__":
    main()
