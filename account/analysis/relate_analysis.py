from collections import Counter, defaultdict
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
import os
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHashLSH, MinHash
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상수 정의
API_KEY = "koba-5JNWNQI-MH5EQJY-QINVNIQ-2IOA5IY"
SIMILARITY_THRESHOLD = 0.8
TOP_KEYWORDS_COUNT = 30
t = brn.Tagger(API_KEY, "localhost")

def load_pretrained_model():
    """
    사전 학습된 한국어 Word2Vec 모델을 로드하는 함수
    """
    try:
        # 모델 파일 경로 설정
        model_path = 'account/static/ko.bin'
        
        if os.path.exists(model_path):
            try:
                # 바이너리 모드로 직접 읽기
                with open(model_path, 'rb') as f:
                    # 첫 4바이트는 매직 넘버일 수 있으므로 건너뛰기
                    magic = f.read(4)
                    logger.info(f"File magic number: {magic.hex()}")
                    
                    # 파일 포맷 확인
                    if magic.startswith(b'\xba\x16O/'):  # 특정 매직 넘버 확인
                        logger.info("파일 포맷 확인됨")
                        # 헤더 정보 읽기
                        header = f.readline().decode('utf-8', errors='ignore').strip()
                        try:
                            vocab_size, vector_size = map(int, header.split())
                            logger.info(f"모델 정보 - 단어 수: {vocab_size}, 벡터 크기: {vector_size}")
                        except ValueError:
                            logger.warning("헤더 파싱 실패, 기본값 사용")
                            vocab_size, vector_size = 100000, 300  # 기본값 설정
                        
                        # KeyedVectors 객체 생성
                        model = KeyedVectors(vector_size)
                        
                        # 단어와 벡터 읽기
                        for _ in range(vocab_size):
                            try:
                                # 단어 읽기
                                word_bytes = bytearray()
                                while True:
                                    b = f.read(1)
                                    if not b or b == b' ':
                                        break
                                    word_bytes.extend(b)
                                
                                if not word_bytes:
                                    break
                                
                                # 단어 디코딩 (여러 인코딩 시도)
                                for encoding in ['utf-8', 'cp949', 'euc-kr']:
                                    try:
                                        word = word_bytes.decode(encoding)
                                        break
                                    except UnicodeDecodeError:
                                        continue
                                else:
                                    word = word_bytes.decode('utf-8', errors='ignore')
                                
                                # 벡터 읽기
                                vector = np.fromfile(f, dtype=np.float32, count=vector_size)
                                if len(vector) != vector_size:
                                    break
                                
                                # 단어와 벡터 추가
                                model.add_vector(word, vector)
                                
                            except Exception as e:
                                logger.error(f"단어 처리 중 오류: {str(e)}")
                                continue
                        
                        return model
                    else:
                        logger.error("알 수 없는 파일 포맷")
                        return None
                        
            except Exception as e:
                logger.error(f"파일 읽기 실패: {str(e)}")
                return None
        else:
            logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return None
            
    except Exception as e:
        logger.error(f"Word2Vec 모델 로드 중 오류 발생: {str(e)}")
        logger.error(f"오류 타입: {type(e)}")
        return None

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
        logger.info("SBERT와 KoSimCSE 모델 로드 완료")
        return models
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")
        return None

# Word2Vec 모델 로드
try:
    logger.info("Word2Vec 모델 로딩 시작...")
    logger.info(f"현재 작업 디렉토리: {os.getcwd()}")
    word2vec_model = load_pretrained_model()
    if word2vec_model is not None:
        logger.info("Word2Vec 모델 로딩 완료")
        logger.info(f"모델 크기: {len(word2vec_model.key_to_index)} 어")
    else:
        logger.warning("Word2Vec 모델 로딩 실패")
except Exception as e:
    logger.error(f"모델 로드 중 예외 발생: {str(e)}")
    word2vec_model = None

# NLP 모델 로드
try:
    nlp_models = load_sbert_model()
except Exception as e:
    logger.error(f"NLP 모델 로드 중 오류: {str(e)}")
    nlp_models = None

# 불용어 로드
def load_stopwords():
    try:
        with open('account/static/불용어.txt', 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    except Exception as e:
        logger.error(f"불용어 파일 로드 실패: {str(e)}")
        return set()

stopwords = load_stopwords()

def extract_keywords_from_desc(text):
    """
    텍스트에서 키워드 추출
    - 형태소 분석 수행
    - 명사만 추출
    - 불용어 제거
    - 중복 제거
    """
    try:
        tagged = t.tags([text])
        keywords = []
        
        for sent in tagged.sentences():
            for token in sent.tokens:
                for morph in token.morphemes:
                    if morph.tag in [24, 25]:  # 일반명사(24)나 고유명사(25)
                        keyword = morph.text.content
                        if isinstance(keyword, bytes):
                            keyword = keyword.decode('utf-8')
                        # 2글자 이상, 불용어 아님, 숫자만으로 구성되지 않음
                        if (len(keyword) > 1 and 
                            keyword not in stopwords and 
                            not keyword.isdigit() and
                            not any(c.isdigit() for c in keyword)):
                            keywords.append(keyword)
        
        # 중복 제거하여 반환
        return list(dict.fromkeys(keywords))
    except Exception as e:
        logger.error(f"키워드 추출 실패: {str(e)}")
        return []

def calculate_similarity(text1, text2):
    """두 텍스트 간 유사도 계산"""
    try:
        if nlp_models is None:
            return 0.0
            
        sbert = nlp_models['sbert']
        embedding1 = sbert.encode([text1])[0]
        embedding2 = sbert.encode([text2])[0]
        
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    except Exception as e:
        logger.error(f"유사도 계산 실패: {str(e)}")
        return 0.0

def calculate_tfidf(documents):
    """TF-IDF 점수 계산"""
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        return vectorizer, tfidf_matrix
    except Exception as e:
        logger.error(f"TF-IDF 계산 실패: {str(e)}")
        return None, None

def create_minhash(text):
    """텍스트에서 MinHash 생성"""
    try:
        # 텍스트를 n-gram으로 분할
        def get_ngrams(text, n=2):
            return [''.join(gram) for gram in zip(*[text[i:] for i in range(n)])]
        
        # MinHash 객체 생성
        m = MinHash(num_perm=128)
        
        # n-gram 생성 및 해싱
        ngrams = get_ngrams(text)
        for gram in ngrams:
            m.update(gram.encode('utf-8'))
        
        return m
    except Exception as e:
        logger.error(f"MinHash 생성 중 오류: {str(e)}")
        return None

def find_similar_keywords_lsh(target_keyword, candidate_keywords, threshold=0.8):
    """LSH를 사용하여 유사한 키워드 찾기"""
    try:
        # LSH 인덱스 생성
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        
        # 후보 키워드들의 MinHash 생성 및 LSH 인덱스에 추가
        keyword_hashes = {}
        for keyword in candidate_keywords:
            m = create_minhash(keyword)
            if m:
                keyword_hashes[keyword] = m
                lsh.insert(keyword, m)
        
        # 타겟 키워드의 MinHash 생성
        target_hash = create_minhash(target_keyword)
        if not target_hash:
            return []
        
        # LSH로 유사한 키워드 찾기
        similar_keywords = lsh.query(target_hash)
        
        # 정확한 유사도 계산 및 필터링
        result = []
        for similar_keyword in similar_keywords:
            if similar_keyword != target_keyword:  # 자기 자신 제외
                similarity = calculate_similarity(target_keyword, similar_keyword)
                if similarity >= threshold:
                    result.append((similar_keyword, similarity))
        
        return sorted(result, key=lambda x: x[1], reverse=True)
        
    except Exception as e:
        logger.error(f"LSH 검색 중 오류: {str(e)}")
        return []

def analyze_related_words(video_desc, video_transcript=None):
    """
    비디오 설명과 자막을 분석하여 연관 단어 네트워크 생성
    """
    try:
        logger.info("연관어 분석 시작")
        
        # 1. 설명에서 모든 키워드 추출
        desc_keywords = extract_keywords_from_desc(video_desc)
        logger.info(f"설명에서 추출된 키워드: {desc_keywords}")
        
        # 디버깅: "윤성"과 "윤석열" 유사도 체크
        test_keywords = ["윤성", "윤석열"]
        sbert = nlp_models['sbert']
        test_embeddings = sbert.encode(test_keywords)
        test_similarity = cosine_similarity([test_embeddings[0]], [test_embeddings[1]])[0][0]
        logger.info(f"디버깅 - '윤성'과 '윤석열'의 유사도: {test_similarity:.4f}")
        
        # 2. 자막 처리 및 상위 30개 키워드 추출
        transcript_keywords = []
        keyword_importance = {}
        if video_transcript:
            # 자막 텍스트 결합
            if isinstance(video_transcript, list):
                transcript_text = ' '.join([item['text'] for item in video_transcript])
            else:
                transcript_text = video_transcript
                
            # 자막에서 키워드 추출
            transcript_keywords = extract_keywords_from_desc(transcript_text)
            
            # TF-IDF 계산
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([' '.join([k] * transcript_keywords.count(k)) for k in set(transcript_keywords)])
            tfidf_scores = dict(zip(set(transcript_keywords), tfidf_matrix.toarray().sum(axis=1)))
            
            # 빈도수 계산
            keyword_freq = Counter(transcript_keywords)
            
            # 중요도 점수 계산 (TF-IDF + 빈도)
            max_freq = max(keyword_freq.values())
            max_tfidf = max(tfidf_scores.values())
            
            for keyword in set(transcript_keywords):
                freq_score = keyword_freq[keyword] / max_freq
                tfidf_score = tfidf_scores[keyword] / max_tfidf
                keyword_importance[keyword] = 0.7 * freq_score + 0.3 * tfidf_score
            
            # 상위 30개 키워드 선정
            transcript_keywords = [k for k, _ in sorted(
                keyword_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:TOP_KEYWORDS_COUNT]]
            
            logger.info(f"자막에서 추출된 상위 {TOP_KEYWORDS_COUNT}개 키워드: {transcript_keywords}")
        
        # 3. LSH를 사용한 효율적인 유사도 계산
        keyword_mapping = {}  # 매핑 결과 저장
        node_sizes = {}       # 노드 크기 저장
        final_keywords = set()  # 최종 키워드 집합
        
        # SBERT 모델을 사용하여 임베딩 계산
        desc_embeddings = sbert.encode(desc_keywords)
        transcript_embeddings = sbert.encode(transcript_keywords)
        
        # 유사도 행렬 계산
        similarity_matrix = cosine_similarity(transcript_embeddings, desc_embeddings)
        
        # 각 자막 키워드에 대해 매핑 수행
        for i, keyword in enumerate(transcript_keywords):
            # 가장 유사한 설명 키워드 찾기
            max_sim_idx = similarity_matrix[i].argmax()
            max_sim = similarity_matrix[i][max_sim_idx]
            
            # 디버깅: 각 키워드의 최대 유사도 출력
            logger.info(f"디버깅 - 키워드 '{keyword}'의 최대 유사도: {max_sim:.4f} (with '{desc_keywords[max_sim_idx]}')")
            
            if max_sim >= SIMILARITY_THRESHOLD:
                # 유사도가 높은 경우 설명 키워드로 매핑
                mapped_keyword = desc_keywords[max_sim_idx]
                keyword_mapping[keyword] = mapped_keyword
                if mapped_keyword not in node_sizes or keyword_importance[keyword] > node_sizes[mapped_keyword]:
                    node_sizes[mapped_keyword] = keyword_importance[keyword]
                final_keywords.add(mapped_keyword)
                logger.info(f"키워드 변환: {keyword} → {mapped_keyword} (유사도: {max_sim:.3f})")
            else:
                # 유사도가 낮은 경우 원본 키워드 유지
                keyword_mapping[keyword] = keyword
                node_sizes[keyword] = keyword_importance[keyword]
                final_keywords.add(keyword)
                logger.info(f"키워드 유지: {keyword}")
        
        # 4. 네트워크 그래프 생성
        G = nx.Graph()
        
        # 노드 추가 (크기는 TF-IDF 기반)
        for keyword in final_keywords:
            G.add_node(keyword, size=node_sizes.get(keyword, 0.5))
            logger.info(f"노드 추가: {keyword} (크기: {node_sizes.get(keyword, 0.5):.3f})")
        
        # 엣지 추가
        keywords = list(final_keywords)
        if keywords:
            # 임베딩 일괄 계산
            embeddings = sbert.encode(keywords)
            similarity_matrix = cosine_similarity(embeddings)
            
            edge_count = 0
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    similarity = similarity_matrix[i][j]
                    if similarity > 0.3:  # 엣지 임계값
                        G.add_edge(keywords[i], keywords[j], weight=similarity)
                        edge_count += 1
                        logger.info(f"엣지 추가: {keywords[i]} - {keywords[j]} (유사도: {similarity:.3f})")
            
            logger.info(f"총 엣지 수: {edge_count}")
        
        # 5. 연관어 쌍 추출
        word_pairs = []
        for u, v, data in G.edges(data=True):
            word_pairs.append(((u, v), data['weight']))
        
        return G, sorted(word_pairs, key=lambda x: x[1], reverse=True)[:20], list(final_keywords)
        
    except Exception as e:
        logger.error(f"연관어 분석 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return nx.Graph(), [], []

def generate_network_graph(G):
    """네트워크 그래프 시각화"""
    try:
        matplotlib.use('Agg')
        plt.clf()  # 기존 그래프 초기화
        
        # 한글 폰트 설정
        try:
            font_path = "C:/Windows/Fonts/malgun.ttf"
            font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        except:
            logger.warning("기본 폰트를 사용합니다.")
            plt.rcParams['font.family'] = 'Malgun Gothic'
        
        plt.figure(figsize=(15, 12))
        
        # 노드 크기 스케일링
        sizes = [G.nodes[node]['size'] for node in G.nodes()]
        if sizes:
            min_size = min(sizes)
            max_size = max(sizes)
            if min_size == max_size:
                scaled_sizes = [5000] * len(sizes)
            else:
                scaled_sizes = [
                    2000 + (size - min_size) * 8000 / (max_size - min_size)
                    for size in sizes
                ]
        else:
            scaled_sizes = [5000] * len(G.nodes())
        
        # 레이아웃 계산 - 더 넓은 공간에 배치
        pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
        
        # 엣지 그리기 - 유사도에 따른 색상 변화
        edge_colors = []
        edge_weights = []
        for (u, v, d) in G.edges(data=True):
            weight = d['weight']
            edge_weights.append(weight * 5)  # 선 두께 증가
            # 유사도에 따른 색상 (파란색 → 빨간색)
            edge_colors.append((weight, 0, 1-weight))
        
        # 엣지 그리기
        nx.draw_networkx_edges(G, pos, 
                             width=edge_weights,
                             edge_color=edge_colors,
                             alpha=0.5)
        
        # 노드 그리기 - 크기에 따른 색상 변화
        node_colors = []
        for node in G.nodes():
            size = G.nodes[node]['size']
            # 크기에 따른 색상 (연한 파랑 → 진한 파랑)
            intensity = (size - min_size) / (max_size - min_size) if max_size != min_size else 0.5
            node_colors.append((0.8-intensity*0.5, 0.8-intensity*0.5, 1))
        
        nx.draw_networkx_nodes(G, pos,
                             node_size=scaled_sizes,
                             node_color=node_colors,
                             alpha=0.6,
                             edgecolors='white',
                             linewidths=2)
        
        # 레이블 그리기 - 노드 크기에 비례하는 폰트 크기
        labels = {node: node for node in G.nodes()}
        font_sizes = {}
        for node in G.nodes():
            size = G.nodes[node]['size']
            if max_size == min_size:
                font_sizes[node] = 12
            else:
                font_sizes[node] = 10 + (size - min_size) * 14 / (max_size - min_size)
        
        # 레이블 배경 추가
        for node, label in labels.items():
            x, y = pos[node]
            plt.text(x, y,
                    label,
                    fontsize=font_sizes[node],
                    fontfamily=plt.rcParams['font.family'],
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(facecolor='white',
                            alpha=0.7,
                            edgecolor='none',
                            boxstyle='round,pad=0.5'))
        
        plt.title('연관어 네트워크', fontdict={'family': plt.rcParams['font.family'], 'size': 20}, pad=20)
        plt.axis('off')
        
        # 여백 조정
        plt.tight_layout(pad=2.0)
        
        # 이미지로 변환 - 더 높은 DPI
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close('all')  # 모든 그래프 창 닫기
        
        return base64.b64encode(image_png).decode('utf-8')
        
    except Exception as e:
        logger.error(f"그래프 생성 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
