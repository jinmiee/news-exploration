from collections import Counter
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
from io import BytesIO
import base64
import bareunpy as brn
import unicodedata
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import urllib.request
import os
from collections import defaultdict
import math
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch

API_KEY = "koba-5JNWNQI-MH5EQJY-QINVNIQ-2IOA5IY"
t = brn.Tagger(API_KEY, "localhost")

# SBERT와 KoSimCSE 모델을 로드하는 함수 정의
def load_sbert_model():
    """
    한국어 SBERT와 KoSimCSE 모델을 로드하는 함수
    """
    try:
        models = {
            'sbert': SentenceTransformer('jhgan/ko-sbert-nli'),
            'simcse': (
                AutoModel.from_pretrained('BM-K/KoSimCSE-roberta'),
                AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
            )
        }
        return models
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        return None

# Word2Vec 모델 로드
try:
    word2vec_model = KeyedVectors.load_word2vec_format('account/static/ko.bin', binary=True)
except Exception as e:
    print(f"Word2Vec 모델 로드 중 오류: {str(e)}")
    word2vec_model = None

# NLP 모델 로드
try:
    nlp_models = load_sbert_model()
except Exception as e:
    print(f"NLP 모델 로드 중 오류: {str(e)}")
    nlp_models = None

def extract_keywords_from_desc(video_desc):
    """
    비디오 설명에서 키워드를 추출하는 함수
    """
    try:
        # 형태소 분석
        tagged = t.tags([video_desc])
        
        # 명사 추출
        keywords = []
        for sent in tagged.sentences():
            for token in sent.tokens:
                for morph in token.morphemes:
                    if morph.tag in [24, 25]:  # 일반명사(24)나 고유명사(25)
                        keyword = morph.text.content
                        if isinstance(keyword, bytes):
                            keyword = keyword.decode('utf-8')
                        if len(keyword) > 1:  # 2글자 이상만 선택
                            keywords.append(keyword)
        
        return keywords
    except Exception as e:
        print(f"키워드 추출 중 오류: {str(e)}")
        return []

def calculate_similarity(text1, text2):
    """
    두 텍스트 간의 유사도를 계산하는 함수
    """
    try:
        if nlp_models is None:
            return 0.0
            
        # SBERT 모델로 임베딩 생성
        sbert = nlp_models['sbert']
        embedding1 = sbert.encode([text1])[0]
        embedding2 = sbert.encode([text2])[0]
        
        # 코사인 유사도 계산
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
            
    except Exception as e:
        print(f"유사도 계산 중 오류: {str(e)}")
        return 0.0

def analyze_related_words(video_desc, video_transcript=None):
    """
    비디오 설명과 자막을 분석하여 연관 단어 네트워크를 생성하는 함수
    """
    try:
        print("연관어 분석 시작...")
        
        # 불용어 로드
        stopwords = set()
        try:
            with open('account/static/불용어.txt', 'r', encoding='utf-8') as f:
                stopwords = set(f.read().splitlines())
        except Exception as e:
            print(f"불용어 파일 로드 중 오류: {str(e)}")
        
        # 자막에서 키워드 추출 및 빈도 계산
        transcript_keywords = []
        keyword_freq = Counter()
        
        if video_transcript:
            transcript_text = ' '.join([item['text'] for item in video_transcript])
            tagged = t.tags([transcript_text])
            
            # 자막에서 명사 추출 (불용어 적용)
            for sent in tagged.sentences():
                for token in sent.tokens:
                    for morph in token.morphemes:
                        if morph.tag in [24, 25]:  # 일반명사(24)나 고유명사(25)
                            keyword = morph.text.content
                            if isinstance(keyword, bytes):
                                keyword = keyword.decode('utf-8')
                            if len(keyword) > 1 and keyword not in stopwords:
                                transcript_keywords.append(keyword)
            
            # 키워드 빈도 계산
            keyword_freq = Counter(transcript_keywords)
            print(f"자막에서 추출된 키워드 빈도: {keyword_freq.most_common(20)}")
            
            # 빈도 기준으로 상위 키워드 선택 (최소 2회 이상 등장)
            important_trans_keywords = [word for word, freq in keyword_freq.items() if freq >= 2]
        else:
            important_trans_keywords = []
        
        # 설명에서 키워드 추출 (불용어 적용)
        desc_keywords = []
        tagged = t.tags([video_desc])
        for sent in tagged.sentences():
            for token in sent.tokens:
                for morph in token.morphemes:
                    if morph.tag in [24, 25]:
                        keyword = morph.text.content
                        if isinstance(keyword, bytes):
                            keyword = keyword.decode('utf-8')
                        if len(keyword) > 1 and keyword not in stopwords:
                            desc_keywords.append(keyword)
        
        print(f"설명에서 추출된 키워드: {desc_keywords}")
        
        # 키워드 매핑 사전 생성 (성능 최적화)
        keyword_mapping = {}
        reverse_mapping = {}  # 설명 키워드를 키로 하는 역방향 매핑
        
        # SBERT 모델로 한 번에 임베딩 계산 (성능 최적화)
        if nlp_models and important_trans_keywords and desc_keywords:
            sbert = nlp_models['sbert']
            trans_embeddings = sbert.encode(important_trans_keywords)
            desc_embeddings = sbert.encode(desc_keywords)
            
            # 유사도 행렬 계산
            similarity_matrix = cosine_similarity(trans_embeddings, desc_embeddings)
            
            # 각 자막 키워드에 대해 가장 유사한 설명 키워드 찾기
            for i, trans_keyword in enumerate(important_trans_keywords):
                max_idx = similarity_matrix[i].argmax()
                max_similarity = similarity_matrix[i][max_idx]
                
                if max_similarity >= 0.8:
                    desc_keyword = desc_keywords[max_idx]
                    print(f"키워드 변환: '{trans_keyword}' → '{desc_keyword}' (유사도: {max_similarity:.3f})")
                    keyword_mapping[trans_keyword] = desc_keyword
                    if desc_keyword not in reverse_mapping:
                        reverse_mapping[desc_keyword] = []
                    reverse_mapping[desc_keyword].append(trans_keyword)
                else:
                    # 자막 키워드들 간의 유사도도 확인
                    trans_similarity = cosine_similarity([trans_embeddings[i]], trans_embeddings)[0]
                    for j, sim in enumerate(trans_similarity):
                        if i != j and sim >= 0.8:
                            other_keyword = important_trans_keywords[j]
                            if other_keyword in keyword_mapping:
                                mapped_keyword = keyword_mapping[other_keyword]
                                keyword_mapping[trans_keyword] = mapped_keyword
                                if mapped_keyword not in reverse_mapping:
                                    reverse_mapping[mapped_keyword] = []
                                reverse_mapping[mapped_keyword].append(trans_keyword)
                                print(f"자막 내 키워드 통합: '{trans_keyword}' → '{mapped_keyword}' (유사도: {sim:.3f})")
                                break
        
        # 매핑된 키워드의 빈도 계산
        keyword_freq_mapped = Counter()
        for k in important_trans_keywords:
            mapped_k = keyword_mapping.get(k, k)
            keyword_freq_mapped[mapped_k] += keyword_freq[k]
            
        # 매핑된 키워드 중 상위 30개 선택 (설명 키워드 우선)
        desc_keywords_set = set(desc_keywords)
        top_mapped_keywords = []
        
        # 1. 먼저 매핑된 설명 키워드를 추가
        for k, freq in keyword_freq_mapped.most_common():
            if len(top_mapped_keywords) >= 30:
                break
            if k in desc_keywords_set or k in reverse_mapping:
                top_mapped_keywords.append(k)
                
        # 2. 나머지 키워드로 채우기
        if len(top_mapped_keywords) < 30:
            for k, freq in keyword_freq_mapped.most_common():
                if len(top_mapped_keywords) >= 30:
                    break
                if k not in top_mapped_keywords:
                    top_mapped_keywords.append(k)
        
        print(f"선택된 상위 매핑 키워드: {top_mapped_keywords}")
        print(f"키워드 매핑 정보: {keyword_mapping}")
        print(f"역방향 매핑 정보: {reverse_mapping}")
        
        # 네트워크 그래프 생성
        G = nx.Graph()
        word_pairs = []
        word_importance = {}
        
        # 먼저 모든 노드를 그래프에 추가
        for word in top_mapped_keywords:
            G.add_node(word)
            # 기본 중요도를 빈도수로 설정
            word_importance[word] = keyword_freq_mapped[word]
        
        # 키워드 쌍 유사도 계산 최적화
        if top_mapped_keywords:
            # 한 번에 임베딩 계산
            sbert = nlp_models['sbert']
            keyword_embeddings = sbert.encode(top_mapped_keywords)
            
            # 유사도 행렬 계산
            similarity_matrix = cosine_similarity(keyword_embeddings)
            
            # 키워드 쌍 생성
            for i in range(len(top_mapped_keywords)):
                for j in range(i + 1, len(top_mapped_keywords)):
                    try:
                        word1 = top_mapped_keywords[i]
                        word2 = top_mapped_keywords[j]
                        
                        if word1 == word2:
                            continue
                        
                        # 기본 유사도
                        similarity = similarity_matrix[i][j]
                        
                        # 빈도를 고려한 가중치 계산
                        freq_weight = (keyword_freq_mapped[word1] + keyword_freq_mapped[word2]) / max(keyword_freq_mapped.values())
                        final_score = similarity * (0.7 + 0.3 * freq_weight)
                        
                        # 임계값을 낮춰서 더 많은 연결 생성
                        if final_score > 0.3:  # 임계값을 0.5에서 0.3으로 낮춤
                            word_pairs.append(((word1, word2), final_score))
                            
                            # 중요도 누적
                            word_importance[word1] = word_importance.get(word1, 0) + final_score * keyword_freq_mapped[word1]
                            word_importance[word2] = word_importance.get(word2, 0) + final_score * keyword_freq_mapped[word2]
                            
                    except Exception as e:
                        print(f"단어 쌍 {word1}-{word2} 처리 중 오류: {str(e)}")
                        continue
        
        # 그래프 생성
        for (word1, word2), score in word_pairs:
            G.add_edge(word1, word2, weight=score)
        
        # 중요 키워드 추출 (빈도수 고려)
        important_keywords = []
        for word, importance in sorted(word_importance.items(), key=lambda x: x[1], reverse=True):
            if len(important_keywords) >= 10:
                break
            if word not in important_keywords:
                important_keywords.append(word)
                print(f"중요 키워드 선택: {word} (중요도: {importance:.3f})")
        
        return G, word_pairs[:20], important_keywords
        
    except Exception as e:
        print(f"연관어 분석 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return nx.Graph(), [], []

def generate_network_graph(G):
    """네트워크 그래프를 생성하는 함수"""
    try:
        # matplotlib 백엔드 설정
        matplotlib.use('Agg')
        
        # 한글 폰트 설정 - Malgun Gothic 사용
        plt.rcParams['font.family'] = 'Malgun Gothic'
        
        # 폰트 경로가 다르다면 직접 지정
        try:
            font_path = "C:/Windows/Fonts/malgun.ttf"
            font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        except:
            print("기본 폰트를 사용합니다.")
        
        # 그래프 크기 설정
        plt.figure(figsize=(8, 6))
        
        # 노드 크기 계산
        degrees = dict(G.degree())
        node_size = [v * 100 for v in degrees.values()]
        
        # 엣지 굵기 계산
        edge_width = [G[u][v].get('weight', 1.0) * 0.3 for u, v in G.edges()]
        
        # 그래프 레이아웃 설정
        pos = nx.spring_layout(G, k=0.8, iterations=50)
        
        # 그래프 그리기
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.4)
        
        # 노드 레이블 설정
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_family=plt.rcParams['font.family'], font_size=8)
        
        # 그래프 제목 설정
        plt.title('연관어 네트워크', fontdict={'family': plt.rcParams['font.family'], 'size': 12}, pad=20)
        plt.axis('off')
        
        # 그래프를 이미지로 변환
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=200, facecolor='white')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return "에러 발생: 그래프 생성 중 오류 발생"