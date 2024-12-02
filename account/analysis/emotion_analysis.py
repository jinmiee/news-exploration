from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from konlpy.tag import Okt
from collections import Counter
import matplotlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
matplotlib.use('Agg')

# OKT 인스턴스를 전역 변수로 생성하여 재사용
okt = Okt()

# 형태소 분석 결과를 캐싱
@lru_cache(maxsize=1000)
def analyze_nouns(text):
    return okt.nouns(text)

def generate_wordcloud(analyzed_comments, max_comments=100):
    # 댓글 수 제한
    comments = analyzed_comments[:max_comments]
    
    # 병렬 처리로 형태소 분석
    with ThreadPoolExecutor() as executor:
        all_nouns = list(executor.map(
            lambda x: analyze_nouns(x.get('comment', '') if isinstance(x, dict) else str(x)), 
            comments
        ))
    
    # 결과 합치기
    words = [noun for nouns in all_nouns for noun in nouns if noun]  # 빈 값 제외
    word_counts = Counter(words)
    
    # 워드클라우드 생성
    font_path = "C:/Windows/Fonts/malgun.ttf"
    wordcloud = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color='white',
        max_words=100
    ).generate_from_frequencies(word_counts)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=200)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

def generate_pie_chart(analyzed_comments):
    # 긍정/부정 개수 계산 최적화
    sentiments = [comment['sentiment'] for comment in analyzed_comments]
    positive_count = sentiments.count('POSITIVE')
    negative_count = sentiments.count('NEGATIVE')
    
    labels = ['긍정', '부정']
    sizes = [positive_count, negative_count]
    colors = ['#66b3ff', '#ff6666']
    explode = (0.1, 0)
    
    # 그래프 크기 최적화
    plt.figure(figsize=(6, 4))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
           autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150)  # dpi 낮춤
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

def analyze_morphemes(analyzed_comments, max_comments=100):
    # 댓글 수 제한
    comments = analyzed_comments[:max_comments]
    
    # 병렬 처리로 형태소 분석
    with ThreadPoolExecutor() as executor:
        all_nouns = list(executor.map(
            lambda x: analyze_nouns(x.get('comment', '') if isinstance(x, dict) else str(x)), 
            comments
        ))
    
    # 결과 합치기
    words = [noun for nouns in all_nouns for noun in nouns if noun]  # 빈 값 제외
    word_counts = Counter(words)
    
    return word_counts.most_common(10)

# 감정 분석 결과를 캐싱
@lru_cache(maxsize=1000)
def analyze_sentiment(comment_text):
    try:
        # 간단한 키워드 기반 감정 분석
        positive_words = ['좋다', '훌륭', '최고', '감사', '멋지', '행복', '사랑', '희망']
        negative_words = ['나쁘', '실망', '최악', '불만', '싫', '화나', '슬프', '걱정']
        
        positive_count = sum(1 for word in positive_words if word in comment_text)
        negative_count = sum(1 for word in negative_words if word in comment_text)
        
        # 감정 결정
        if positive_count > negative_count:
            sentiment = 'POSITIVE'
            confidence = 0.6 + (0.4 * (positive_count / (positive_count + negative_count)))
        elif negative_count > positive_count:
            sentiment = 'NEGATIVE'
            confidence = 0.6 + (0.4 * (negative_count / (positive_count + negative_count)))
        else:
            # 중립이면 약간 긍정적으로 처리
            sentiment = 'POSITIVE'
            confidence = 0.6
        
        return {
            'comment': comment_text,
            'sentiment': sentiment,
            'confidence': confidence
        }
    except Exception as e:
        print(f"감정 분석 오류: {e}")
        # 오류 발생시 기본값 반환
        return {
            'comment': comment_text,
            'sentiment': 'POSITIVE',
            'confidence': 0.6
        } 