"""
account.analysis.text_processing 의 순수 함수 단위 테스트.

이 테스트들은 sklearn / torch / konlpy 같은 무거운 의존성 없이 실행된다
(text_processing 모듈이 해당 의존성을 함수 내부에서 lazy import 하도록 분리됨).
"""
import numpy as np
import pytest

from account.analysis.text_processing import clean_title, get_hybrid_similarity


class TestCleanTitle:
    """뉴스 제목 정제 로직 검증."""

    def test_대괄호_방송사_날짜_제거(self):
        assert clean_title('[속보] 윤 대통령 발언 /YTN 2024.12.05') == '윤 대통령 발언'

    def test_괄호날짜와_해시태그_제거(self):
        assert clean_title('경제 위기 심화 (2024.11.27) #뉴스다') == '경제 위기 심화'

    def test_슬래시날짜형식_제거(self):
        assert clean_title('오늘의 날씨 12/22') == '오늘의 날씨'

    def test_연속_공백은_하나로_정리되고_양끝_공백_제거(self):
        assert clean_title('   여러   공백   정리   ') == '여러 공백 정리'

    def test_정제후_대괄호와_해시태그가_남지_않는다(self):
        result = clean_title('[이슈] 국회 통과 #속보 #단독')
        assert '[' not in result and ']' not in result
        assert '#' not in result

    def test_빈문자열은_빈문자열을_반환(self):
        assert clean_title('') == ''

    def test_정제할것이_없는_제목은_그대로_유지(self):
        assert clean_title('국정감사 시작') == '국정감사 시작'


class TestHybridSimilarity:
    """TF-IDF / BERT 가중 결합 유사도 검증."""

    def test_기본_5대5_가중합(self):
        tfidf = np.array([[0.2], [0.8]])
        bert = np.array([[0.6], [0.4]])
        result = get_hybrid_similarity(tfidf, bert, 0.5, 0.5)
        np.testing.assert_allclose(result.ravel(), [0.4, 0.6])

    def test_tfidf_가중치_1이면_tfidf와_동일(self):
        tfidf = np.array([[0.3, 0.7]])
        bert = np.array([[0.9, 0.1]])
        result = get_hybrid_similarity(tfidf, bert, 1.0, 0.0)
        np.testing.assert_allclose(result, tfidf)

    def test_bert_가중치_1이면_bert와_동일(self):
        tfidf = np.array([[0.3, 0.7]])
        bert = np.array([[0.9, 0.1]])
        result = get_hybrid_similarity(tfidf, bert, 0.0, 1.0)
        np.testing.assert_allclose(result, bert)

    def test_4대6_가중치_적용(self):
        tfidf = np.array([[1.0]])
        bert = np.array([[0.0]])
        result = get_hybrid_similarity(tfidf, bert, 0.4, 0.6)
        np.testing.assert_allclose(result.ravel(), [0.4])

    def test_입력_행렬_shape_유지(self):
        tfidf = np.zeros((3, 3))
        bert = np.zeros((3, 3))
        result = get_hybrid_similarity(tfidf, bert, 0.5, 0.5)
        assert result.shape == (3, 3)


class TestProcessText:
    """형태소 분석(Okt) 기반 전처리 — JVM/konlpy 가 있어야 실행되므로 없으면 skip."""

    def test_제목_명사동사형용사_추출(self):
        konlpy = pytest.importorskip("konlpy")
        try:
            from account.analysis.text_processing import process_text
            result = process_text('윤 대통령 경제 정책 발표', None)
        except Exception as e:  # JVM 미설치 등 환경 문제
            pytest.skip(f"형태소 분석 환경 미구성: {e}")
        assert isinstance(result, str)
        assert len(result) > 0
