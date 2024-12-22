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
from django.shortcuts import render
from ..models import YouTubeData
from django.db.models import Q

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상수 정의
API_KEY = "koba-5JNWNQI-MH5EQJY-QINVNIQ-2IOA5IY"
SIMILARITY_THRESHOLD = 0.73
TOP_KEYWORDS_COUNT = 30
t = brn.Tagger(API_KEY, "localhost")

# 파일 상단에 추가
import logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)  # INFO 대신 WARNING으로 설정

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
    # logger.info(f"현재 작업 디렉토리: {os.getcwd()}")
    word2vec_model = load_pretrained_model()
    if word2vec_model is not None:
        logger.info("Word2Vec 모델 로딩 완료")
        # logger.info(f"모델 크기: {len(word2vec_model.key_to_index)} ")
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

def analyze_related_words(video_desc, video_transcript, clean_title_func=None):
    """
    비디오 설명과 자막을 분석하여 연관 단어 네트워크 생성
    """
    try:
        logger.info("연관어 분석 시작")
        
        # clean_title_func가 전달되지 않았다면 원본 텍스트 사용
        if clean_title_func:
            cleaned_desc = clean_title_func(video_desc)
        else:
            cleaned_desc = video_desc

        # 특정 키워드 매핑 정의
        special_mappings = {
            "윤성": "윤석열",
            "윤성열": "윤석열"
        }
        
        # 1. 설명에서 모든 키워드 추출
        desc_keywords = extract_keywords_from_desc(cleaned_desc)
        logger.info(f"설명에서 추출된 키워드: {desc_keywords}")
        
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
            all_transcript_keywords = extract_keywords_from_desc(transcript_text)
            
            if not all_transcript_keywords:
                logger.warning("자막에서 추출된 키워드가 없습니다.")
                return nx.Graph(), [], []
            
            # TF-IDF 계산을 위한 문서 생성
            documents = [' '.join([k] * all_transcript_keywords.count(k)) for k in set(all_transcript_keywords)]
            if not documents:
                logger.warning("TF-IDF 계산을 위한 문서가 없습니다.")
                return nx.Graph(), [], []
                
            # TF-IDF 계산
            try:
                vectorizer = TfidfVectorizer(min_df=1)  # 최소 문서 빈도를 1로 설정
                tfidf_matrix = vectorizer.fit_transform(documents)
                tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray().sum(axis=0)))
            except Exception as e:
                logger.error(f"TF-IDF 계산 중 오류: {str(e)}")
                return nx.Graph(), [], []
            
            # 빈도수 계산
            keyword_freq = Counter(all_transcript_keywords)
            
            # 중요도 점수 계산 (TF-IDF + 빈도)
            max_freq = max(keyword_freq.values())
            max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1.0
            
            for keyword in set(all_transcript_keywords):
                freq_score = keyword_freq[keyword] / max_freq
                tfidf_score = tfidf_scores.get(keyword, 0) / max_tfidf
                keyword_importance[keyword] = 0.7 * freq_score + 0.3 * tfidf_score
            
            # 상위 30개 키워드 선정
            transcript_keywords = [k for k, _ in sorted(
                keyword_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:TOP_KEYWORDS_COUNT]]
            
            logger.info(f"자막에서 추출된 상위 {TOP_KEYWORDS_COUNT}개 키워드: {transcript_keywords}")
        
        if not transcript_keywords:
            logger.warning("처리할 키워드가 없습니다.")
            return nx.Graph(), [], []
            
        # 3. LSH, SBERT, SimCSE 결합한 효율적인 유사도 계산
        keyword_mapping = {}  # 매핑 결과 저장
        node_sizes = {}       # 노드 크기 저장
        final_keywords = set()  # 최종 키워드 집합
        
        if desc_keywords and transcript_keywords:
            # LSH 인덱스 생성
            lsh = MinHashLSH(threshold=0.5, num_perm=128)
            desc_minhashes = {}
            
            # 설명 키워드의 MinHash 생성 및 LSH 인덱스에 추가
            for keyword in desc_keywords:
                m = create_minhash(keyword)
                if m:
                    desc_minhashes[keyword] = m
                    lsh.insert(keyword, m)
            
            # SBERT와 SimCSE 임베딩 미리 계산
            sbert = nlp_models['sbert']
            simcse_model, simcse_tokenizer = nlp_models['simcse']
            
            desc_sbert_embeddings = sbert.encode(desc_keywords)
            desc_inputs = simcse_tokenizer(desc_keywords, padding=True, truncation=True, return_tensors="pt")
            desc_simcse_embeddings = simcse_model(**desc_inputs).pooler_output.detach().numpy()
            
            # 각 자막 키워드에 대해 매핑 수행
            for keyword in transcript_keywords:
                # 특정 키워드 매핑 체크
                if keyword in special_mappings:
                    mapped_keyword = special_mappings[keyword]
                    if mapped_keyword in desc_keywords:
                        keyword_mapping[keyword] = mapped_keyword
                        node_sizes[mapped_keyword] = keyword_importance[keyword]
                        final_keywords.add(mapped_keyword)
                        continue
                
                # LSH로 후보 키워드 추출
                m = create_minhash(keyword)
                if m:
                    candidates = lsh.query(m)
                    if candidates:
                        # 후보 키워드에 대해서만 SBERT와 SimCSE 유사도 계산
                        keyword_sbert_embedding = sbert.encode([keyword])[0]
                        keyword_inputs = simcse_tokenizer([keyword], padding=True, truncation=True, return_tensors="pt")
                        keyword_simcse_embedding = simcse_model(**keyword_inputs).pooler_output.detach().numpy()[0]
                        
                        max_sim = 0
                        best_match = None
                        
                        for candidate in candidates:
                            idx = desc_keywords.index(candidate)
                            # SBERT와 SimCSE 유사도 결합
                            sbert_sim = cosine_similarity([keyword_sbert_embedding], [desc_sbert_embeddings[idx]])[0][0]
                            simcse_sim = cosine_similarity([keyword_simcse_embedding], [desc_simcse_embeddings[idx]])[0][0]
                            combined_sim = (sbert_sim + simcse_sim) / 2
                            
                            if combined_sim > max_sim:
                                max_sim = combined_sim
                                best_match = candidate
                        
                        if max_sim >= SIMILARITY_THRESHOLD:
                            keyword_mapping[keyword] = best_match
                            node_sizes[best_match] = keyword_importance[keyword]
                            final_keywords.add(best_match)
                            continue
                
                # 매핑되지 않은 경우 원본 유지
                keyword_mapping[keyword] = keyword
                node_sizes[keyword] = keyword_importance[keyword]
                final_keywords.add(keyword)
        
        # 4. 네트워크 그래프 생성
        G = nx.Graph()
        
        # 노드 추가 (크기는 TF-IDF 기반)
        for keyword in final_keywords:
            G.add_node(keyword, size=node_sizes[keyword])
        
        # 엣지 추가
        keywords = list(final_keywords)
        if len(keywords) > 1:
            # 임베딩 일괄 계산
            embeddings = nlp_models['sbert'].encode(keywords)
            similarity_matrix = cosine_similarity(embeddings)
            
            # 중심성이 가장 높은 노드 찾기
            edge_weights = {}
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    similarity = similarity_matrix[i][j]
                    if similarity > 0.3:  # 엣지 임계값
                        # TF-IDF 가중치를 반영한 엣지 가중치 계산
                        weight = similarity * (node_sizes[keywords[i]] + node_sizes[keywords[j]]) / 2
                        edge_weights[(keywords[i], keywords[j])] = weight
                        G.add_edge(keywords[i], keywords[j], weight=weight)
            
            # 중심성 계산
            centrality = nx.degree_centrality(G)
            central_node = max(centrality.items(), key=lambda x: x[1])[0]
            
            # 중심 노드와의 거리에 따라 레이아웃 조정을 위해 속성 추가
            distances = nx.single_source_shortest_path_length(G, central_node)
            nx.set_node_attributes(G, distances, 'distance_from_center')
        
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
        # 그래프가 비어있는지 확인
        if len(G.nodes()) == 0:
            logger.warning("그래프에 노드가 없습니다.")
            return None
            
        matplotlib.use('Agg')
        plt.clf()
        
        # 한글 폰트 설정
        try:
            font_path = "C:/Windows/Fonts/malgun.ttf"
            font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        except:
            logger.warning("기본 폰트를 사용합니다.")
            plt.rcParams['font.family'] = 'Malgun Gothic'
        
        plt.figure(figsize=(16, 12), facecolor='white')
        
        # 노드 크기와 색상 계산
        node_sizes = nx.get_node_attributes(G, 'size')
        if not node_sizes:
            logger.warning("노드 크기 정보가 없습니다.")
            return None
            
        min_size = min(node_sizes.values())
        max_size = max(node_sizes.values())
        
        # min_size와 max_size가 같은 경우 처리
        if min_size == max_size:
            scaled_sizes = {node: 7500 for node in G.nodes()}
            node_colors = {node: plt.cm.Blues(0.75) for node in G.nodes()}
        else:
            # 노드 크기 스케일링 (더 넓은 범위로)
            scaled_sizes = {
                node: 3000 + (node_sizes[node] - min_size) * 12000 / (max_size - min_size)
                for node in G.nodes()
            }
            
            # 노드 색상 계산 (중요도에 따라 파란색 계열로 그라데이션)
            node_colors = {
                node: plt.cm.Blues(0.5 + 0.5 * (node_sizes[node] - min_size) / (max_size - min_size))
                for node in G.nodes()
            }
        
        # 중심성 기반 레이아웃 계산
        if len(G.nodes()) > 1 and 'distance_from_center' in G.nodes[list(G.nodes())[0]]:
            # 방사형 레이아웃 생성 (더 넓은 공간 활용)
            pos = {}
            max_distance = max(nx.get_node_attributes(G, 'distance_from_center').values())
            central_node = min(G.nodes(), key=lambda n: G.nodes[n]['distance_from_center'])
            
            # 중심 노드는 가운데에 배치
            pos[central_node] = (0, 0)
            
            # 나머지 노드들을 방사형으로 배치
            other_nodes = [n for n in G.nodes() if n != central_node]
            if other_nodes:  # 다른 노드가 있는 경우에만 각도 계산
                angles = np.linspace(0, 2*np.pi, len(other_nodes), endpoint=False)
                
                for node, angle in zip(other_nodes, angles):
                    distance = G.nodes[node]['distance_from_center']
                    radius = 0.3 + (distance / max_distance) * 0.7  # 더 넓은 반경 사용
                    pos[node] = (
                        radius * np.cos(angle),
                        radius * np.sin(angle)
                    )
        else:
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # 엣지 그리기 (가중치에 따른 색상과 두께)
        edge_weights = nx.get_edge_attributes(G, 'weight')
        if edge_weights:  # 엣지가 있는 경우에만 처리
            min_weight = min(edge_weights.values())
            max_weight = max(edge_weights.values())
            
            # 엣지를 가중치 순으로 정렬하여 그리기 (중요한 연결이 위에 오도록)
            sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
            
            for u, v, data in sorted_edges:
                weight = data['weight']
                # 가중치에 따른 색상 (연한 회색 → 진한 회색)
                if min_weight == max_weight:
                    alpha = 0.6
                    width = 4
                else:
                    alpha = 0.3 + 0.6 * (weight - min_weight) / (max_weight - min_weight)
                    width = 1 + 8 * (weight - min_weight) / (max_weight - min_weight)
                
                nx.draw_networkx_edges(G, pos,
                                     edgelist=[(u, v)],
                                     width=width,
                                     edge_color='gray',
                                     alpha=alpha,
                                     style='solid')
        
        # 노드 그리기
        for node in G.nodes():
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=[node],
                                 node_size=scaled_sizes[node],
                                 node_color=[node_colors[node]],
                                 alpha=0.9,
                                 edgecolors='white',
                                 linewidths=2)
        
        # 레이블 그리기 (크기에 따른 폰트 크기 차등)
        for node in G.nodes():
            x, y = pos[node]
            size = scaled_sizes[node]
            # 노드 크기에 비례하는 폰트 크기 (더 큰 차이)
            fontsize = 12 + (size - 3000) * 16 / 12000
            
            # 중심 노드는 더 강조
            if 'distance_from_center' in G.nodes[node] and G.nodes[node]['distance_from_center'] == 0:
                fontsize *= 1.2
                weight = 'bold'
            else:
                weight = 'normal'
            
            plt.text(x, y,
                    node,
                    fontsize=fontsize,
                    fontweight=weight,
                    fontfamily=plt.rcParams['font.family'],
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(facecolor='white',
                            alpha=0.8,
                            edgecolor='none',
                            boxstyle='round,pad=0.5'))
        
        plt.title('연관어 네트워크', fontdict={'family': plt.rcParams['font.family'], 'size': 24, 'weight': 'bold'}, pad=20)
        plt.axis('off')
        
        # 여백 조정
        plt.tight_layout(pad=1.5)
        
        # 고품질 이미지 저장
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close('all')
        
        return base64.b64encode(image_png).decode('utf-8')
        
    except Exception as e:
        logger.error(f"그래프 생성 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def relate(request):
    video_url = request.GET.get('url')
    video_id = request.GET.get('id')
    
    video = YouTubeData.objects.filter(url=video_url).first()
    
    if video and video.transcript:
        try:
            # 함수 내부에서 동적으로 import
            from ..views import clean_title
            
            # 제목과 설명 불용어 처리
            cleaned_title = clean_title(video.title)
            video_desc = f"{cleaned_title} {video.desc if video.desc else ''}"
            
            # transcript 데이터를 시간 단위로 구분하여 텍스트로 변환
            transcript_segments = []
            for item in video.transcript:
                if 'start' in item and 'text' in item:
                    start_time = int(float(item['start']))
                    minutes = start_time // 60
                    seconds = start_time % 60
                    time_str = f"{minutes:02d}:{seconds:02d}"
                    transcript_segments.append({
                        'time': time_str,
                        'text': item['text']
                    })
            
            # clean_title 함수를 파라미터로 전달
            graph, top_pairs, important_keywords = analyze_related_words(
                video_desc, 
                video.transcript,
                clean_title_func=clean_title
            )
            
            network_graph = generate_network_graph(graph)
            
            # 키워드별 관련 뉴스 분류 추가
            categorized_news = {}
            if important_keywords:
                for keyword in important_keywords:
                    related_news = YouTubeData.objects.filter(
                        Q(title__icontains=keyword) | 
                        Q(desc__icontains=keyword)
                    ).exclude(url=video_url)[:6]
                    
                    if related_news:
                        cleaned_news = []
                        for news in related_news:
                            news.title = clean_title(news.title)
                            try:
                                news.video_id = news.url.split('v=')[1].split('&')[0]
                            except:
                                news.video_id = None
                            cleaned_news.append(news)
                        categorized_news[keyword] = cleaned_news
            
            context = {
                'section': 'relate',
                'video': video,
                'video_title': cleaned_title,
                'network_graph': network_graph,
                'top_pairs': top_pairs,
                'categorized_news': categorized_news,
                'important_keywords': important_keywords,
                'transcript_segments': transcript_segments
            }
        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            context = {
                'section': 'relate',
                'error_message': '분석 중 오류가 발생했습니다.'
            }
    else:
        context = {
            'section': 'relate',
            'error_message': '비디오 자막이 없거나 비디오를 찾을 수 없습니다.'
        }
    
    return render(request, 'analysis/relate.html', context)
