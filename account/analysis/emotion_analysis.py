from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from konlpy.tag import Okt
from collections import Counter
import matplotlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from transformers import pipeline

matplotlib.use('Agg')

# 전역 변수로 설정하여 재사용
okt = Okt()
sentiment_analysis_pipeline = pipeline("sentiment-analysis", truncation=True, padding=True, max_length=512)


@lru_cache(maxsize=1000)
def analyze_nouns(text):
    return okt.nouns(text)


@lru_cache(maxsize=1000)
def analyze_sentiment(comment_text):
    try:
        result = sentiment_analysis_pipeline(comment_text)[0]
        return {
            'comment': comment_text,
            'sentiment': result['label'],
            'confidence': result['score']
        }
    except Exception as e:
        print(f"감정 분석 오류: {e}")
        return {
            'comment': comment_text,
            'sentiment': 'POSITIVE',
            'confidence': 0.6
        }


def generate_wordcloud(comments, analyzed_comments, max_comments=100):

    # 한글 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 환경
    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rc('font', family=font_name)

    try:
        # 댓글 수 제한
        limited_comments = comments[:max_comments]

        # 색상 함수 정의
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            for sentiment in analyzed_comments:
                if sentiment['comment'] and word in sentiment['comment']:
                    return 'rgb(0, 0, 255)' if sentiment['sentiment'] == 'POSITIVE' else 'rgb(255, 0, 0)'
            return 'rgb(0, 0, 0)'

        # 병렬 처리로 형태소 분석
        with ThreadPoolExecutor() as executor:
            all_nouns = list(executor.map(analyze_nouns, limited_comments))

        # 결과 합치기
        text = " ".join([noun for nouns in all_nouns for noun in nouns if noun])

        if not text.strip():
            raise ValueError("No words available to generate WordCloud.")

        # 워드클라우드 생성
        font_path = "C:/Windows/Fonts/malgun.ttf"  # 시스템에 맞는 폰트 경로로 변경
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            font_path=font_path,
            color_func=color_func,
            max_words=100
        ).generate(text)

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
    except Exception as e:
        print(f"워드클라우드 생성 오류: {e}")
        return None


def generate_pie_chart(analyzed_comments):
    try:
        # 긍정/부정 개수 계산 최적화
        sentiments = [comment['sentiment'] for comment in analyzed_comments]
        positive_count = sentiments.count('POSITIVE')
        negative_count = sentiments.count('NEGATIVE')

        labels = ['긍정', '부정']
        sizes = [positive_count, negative_count]
        colors = ['#66b3ff', '#ff6666']
        explode = (0.1, 0)

        # 도넛 차트로 변경: 'wedgeprops' 옵션을 추가하여 구멍을 뚫음
        plt.figure(figsize=(6, 4))
        wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='%1.1f%%', shadow=True, startangle=140, wedgeprops={'width': 0.5})
        plt.axis('equal')

        # 글자 크기 설정
        for autotext in autotexts:
            autotext.set_fontsize(12)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        print(f"파이차트 생성 오류: {e}")
        return None


def analyze_morphemes(analyzed_comments, max_comments=100):
    try:
        # 댓글 수 제한
        comments = [comment['comment'] for comment in analyzed_comments[:max_comments]]

        # 병렬 처리로 형태소 분석
        with ThreadPoolExecutor() as executor:
            all_nouns = list(executor.map(analyze_nouns, comments))

        # 결과 합치기
        words = [noun for nouns in all_nouns for noun in nouns if noun]
        word_counts = Counter(words)

        return word_counts.most_common(10)
    except Exception as e:
        print(f"형태소 분석 오류: {e}")
        return []
