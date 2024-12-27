from datetime import datetime, timedelta
from ..models import RelatedWordAnalysis, Chart
from ..analysis.relate_analysis import analyze_related_words
from ..analysis.text_processing import clean_title
from ..analysis.visualization import generate_network_graph
import traceback

def save_related_word_analysis():
    """
    Chart에 있는 뉴스들의 연관어 분석을 수행하고 결과를 저장
    """
    try:
        print("연관어 분석 작업 시작...")
        # Chart에서 뉴스 가져오기
        chart_videos = Chart.objects.all().order_by('rank')
        video_count = chart_videos.count()
        
        print(f"분석 대상 비디오 수: {video_count}")
        
        if video_count == 0:
            print("분석할 비디오가 없습니다.")
            return
            
        processed_count = 0
        for video in chart_videos:
            try:
                processed_count += 1
                print(f"처리 중: {processed_count}/{video_count} ({(processed_count/video_count)*100:.1f}%)")
                
                if not video.transcript:
                    print(f"자막이 없는 비디오 건너뛰기: {video.url}")
                    continue

                if RelatedWordAnalysis.objects.filter(video_id=video.url).exists():
                    print(f"이미 분석된 비디오 건너뛰기: {video.url}")
                    continue

                print(f"비디오 분석 시작: {video.url}")
                # 제목과 설명 불용어 처리
                cleaned_title = clean_title(video.title)
                video_desc = f"{cleaned_title} {video.desc if video.desc else ''}"
                
                # 연관어 분석 수행
                graph, top_pairs, important_keywords, _ = analyze_related_words(
                    video_desc,
                    video.transcript,
                    clean_title_func=clean_title
                )

                # 분석 결과 저장
                RelatedWordAnalysis.objects.create(
                    video_id=video.url,
                    network_graph=generate_network_graph(graph),
                    top_pairs=top_pairs,
                    important_keywords=important_keywords
                )
                
                print(f"분석 결과 저장 완료: {video.url}")
                print(f"저장된 데이터 확인: {RelatedWordAnalysis.objects.get(video_id=video.url).important_keywords}")

            except Exception as e:
                print(f"비디오 {video.url} 분석 중 오류 발생: {str(e)}")
                print(traceback.format_exc())
                continue

    except Exception as e:
        print(f"연관어 분석 태스크 실행 중 오류: {str(e)}") 