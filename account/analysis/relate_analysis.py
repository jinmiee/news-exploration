import platform
from collections import Counter, defaultdict
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from io import BytesIO
import base64
import bareunpy as brn
import unicodedata
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from umap import UMAP
import os
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHashLSH, MinHash
import re
from django.shortcuts import render
from ..models import YouTubeData, RelatedWordAnalysis, Chart
from django.db.models import Q
from time import time
import pandas as pd
from .visualization import generate_network_graph
import psutil
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def relate(request):
    video_url = request.GET.get('url')
    video = Chart.objects.filter(url=video_url).first()
    
    if video and video.transcript:
        try:
            # 1. 먼저 저장된 분석 결과가 있는지 확인
            analysis = RelatedWordAnalysis.objects.filter(video_id=video.url).first()
            
            if analysis:
                # 2A. 기존 분석 결과가 있으면 그대로 사용
                context = {
                    'section': 'relate',
                    'video': video,
                    'video_title': clean_title(video.title),
                    'network_graph': analysis.network_graph,
                    'top_pairs': analysis.top_pairs,
                    'important_keywords': analysis.important_keywords,
                    'transcript_segments': get_transcript_segments(video.transcript)
                }
            else:
                # 2B. 없으면 실시간으로 분석 수행
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
                graph, top_pairs, important_keywords, performance_graph = analyze_related_words(
                    video_desc,
                    video.transcript,
                    clean_title_func=clean_title
                )

                network_graph = generate_network_graph(graph)

                # 분석 결과 저장
                RelatedWordAnalysis.objects.create(
                    video_id=video.url,
                    network_graph=network_graph,
                    top_pairs=top_pairs,
                    important_keywords=important_keywords
                )

                context = {
                    'section': 'relate',
                    'video': video,
                    'video_title': cleaned_title,
                    'network_graph': network_graph,
                    'silhouette_graph': performance_graph,
                    'top_pairs': top_pairs,
                    'important_keywords': important_keywords,
                    'transcript_segments': transcript_segments
                }

            return render(request, 'analysis/relate.html', context)
            
        except Exception as e:
            logger.error(f"분석 중 오류 발생: {str(e)}")
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



# 로깅 설정
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 추가 로깅 설정
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('gensim').setLevel(logging.ERROR)

# 상수 정의
API_KEY = os.getenv("YOUTUBE_API_KEY")
SIMILARITY_THRESHOLD = 0.73
EDGE_THRESHOLD = 0.5
TOP_KEYWORDS_COUNT = 20
t = brn.Tagger(API_KEY, "localhost")
COMMON_WORDS = {
    '오늘', '관련', '가능', '당일', '이번', '현재', '지금', '계획', '예정',
    '진행', '추후', '방향', '방침', '대상', '자체', '과정', '입장', '일체',
    '관계', '정도', '상황', '결과', '내용', '의견', '사실', '기준', '이후',
    '최근', '기존', '향후', '대부분', '일부', '전체', '기타', '가운데', '간주'
}

# 파일 상단에 전역 변수로 SBERT 모델 로드 (한 번만 로드되도록)
_nlp_models = None

def get_nlp_models():
    global _nlp_models
    if _nlp_models is None:
        try:
            _nlp_models = {
                'sbert': SentenceTransformer('jhgan/ko-sbert-nli', device='cpu'),
                'simcse': (
                    AutoModel.from_pretrained('BM-K/KoSimCSE-roberta', revision='main', token=None),
                    AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta', revision='main', token=None)
                )
            }
            logger.info("NLP 모델 로드 완료")
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {str(e)}")
            return None
    return _nlp_models

# 불용어 로드
def load_stopwords():
    """불용어 사전 로드"""
    try:
        with open('account/static/불용어.txt', 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    except Exception as e:
        logger.error(f"불용어 파일 로드 실패: {str(e)}")
        return set()

# 전역 변수 불용어 로드
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
                        # 키워드 필터링 조건 강화
                        if (len(keyword) > 1 and  # 2글자 이상
                            keyword not in stopwords and  # 불용어 아님
                            not keyword.isdigit() and  # 숫자만로 구성되지 않음
                            not any(c.isdigit() for c in keyword) and  # 숫자 포함하지 않음
                            not any(c.isspace() for c in keyword) and  # 공백 포함하지 않음
                            not any(c in '습니다.,' for c in keyword) and  # 조사/어미 제외
                            not keyword.endswith(('는', '을', '를', '이', '가', '의', '로', '에', '도'))):  # 조사로 끝나지 않음
                            keywords.append(keyword)
        
        # 중복 제거하여 반환
        return list(dict.fromkeys(keywords))
    except Exception as e:
        logger.error(f"키워드 추출 실패: {str(e)}")
        return []

def calculate_similarity(text1, text2):
    """두 텍스트 간 유사도 계산"""
    try:
        models = get_nlp_models()
        if models is None:
            return 0.0
            
        sbert = models['sbert']
        # 배치 처리로 변경하여 효율성 향상
        embeddings = sbert.encode([text1, text2])
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    except Exception as e:
        logger.error(f"유사도 계산 실패: {str(e)}")
        return 0.0

def calculate_tfidf(documents):
    """TF-IDF 점수 계산"""
    try:
        vectorizer = TfidfVectorizer(min_df=1, random_state=42)
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

def analyze_related_words(video_desc, transcript, clean_title_func=None):
    try:
        logger.info("\n[연관어 분석 시작]")
        
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
        logger.info(f"[설명 키워드] 추출된 키워드 ({len(desc_keywords)}개): {desc_keywords}")
        
        # 2. 자막 처리 및 상위 키워드 추출
        transcript_keywords = []
        keyword_importance = {}
        if transcript:
            # 자막 텍스트 결합
            if isinstance(transcript, list):
                transcript_text = ' '.join([item['text'] for item in transcript])
            else:
                transcript_text = transcript
                
            # 자막에서 키워드 추출
            all_transcript_keywords = extract_keywords_from_desc(transcript_text)
            logger.info(f"[자막] 전체 추출 키워드 수: {len(all_transcript_keywords)}개")
            logger.info(f"[자막] 추출된 전체 키워드: {all_transcript_keywords}")  # 디버깅용
            
            if not all_transcript_keywords:
                logger.info("[경고] 자막에서 추출된 키워드가 없습니다")
                return nx.Graph(), [], [], None
            
            # TF-IDF 산 부분 수정
            try:
                # 문서 생성 방식 변경
                documents = []
                for sentence in transcript_text.split('.'):
                    if sentence.strip():
                        documents.append(sentence.strip())
                
                # TF-IDF 계
                vectorizer = TfidfVectorizer(min_df=1)
                tfidf_matrix = vectorizer.fit_transform(documents)
                
                # 각 단어별 TF-IDF 점수 계산 (문서 전체에서의 중요도)
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = {}
                for keyword in set(all_transcript_keywords):
                    if keyword in feature_names:
                        idx = feature_names.tolist().index(keyword)
                        # 모든 문서에서의 TF-IDF 점수 평균
                        tfidf_scores[keyword] = np.mean(tfidf_matrix[:, idx].toarray())
                
                # 빈도수 계산
                keyword_freq = Counter(all_transcript_keywords)
                
                # 정규화를 위한 최대값 계산
                max_freq = max(keyword_freq.values())
                max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1.0
                
                # 중요도 점수 계산
                keyword_importance = {}
                for keyword in set(all_transcript_keywords):
                    # 빈도수 점수 (0~1)
                    freq_score = keyword_freq[keyword] / max_freq
                    
                    # TF-IDF 점수 (0~1)
                    tfidf_score = tfidf_scores.get(keyword, 0) / max_tfidf
                    
                    # 기본 가중치 적용 (40:60)
                    weighted_score = 0.4 * freq_score + 0.6 * tfidf_score
                    
                    # 가중치 보정
                    # 1. 제목/설명에 있는 키워드는 가중치 크게 증가
                    if keyword in desc_keywords:
                        weighted_score *= 2.0
                    
                    # 2. 일반적인 단어는 가중치 폭 감소
                    if keyword in COMMON_WORDS:
                        weighted_score *= 0.3
                    
                    # 3. 2음절 이하의 단어는 가중치 감소 (제목/설명에 있는 경우 제외)
                    if len(keyword) <= 2 and keyword not in desc_keywords:
                        weighted_score *= 0.4
                    
                    # # 4. 고유명사나 특정 주제어는 가중치 증가
                    # if any(topic in keyword for topic in ['탄', '대통령', '윤석열', '민주당', '국회']):
                    #     weighted_score *= 1.8
                    
                    # 5. TF-IDF 점수가 빈도수보다 현저히 높은 경우 (문맥적 중요성)
                    if tfidf_score > freq_score * 2:
                        weighted_score *= 1.5
                    
                    # 6. 숫자나 특수자가 포함된 경우 가중치 감소
                    if any(c.isdigit() or not c.isalnum() for c in keyword):
                        weighted_score *= 0.5
                    
                    keyword_importance[keyword] = weighted_score
                
                # 상위 25개 키워드 선정
                transcript_keywords = [k[0] for k in sorted(
                    keyword_importance.items(),
                    key=lambda x: (-x[1], x[0])
                )[:TOP_KEYWORDS_COUNT]]
                
                logger.info(f"[자막] 선정된 상위 {TOP_KEYWORDS_COUNT}개 키워드: {transcript_keywords}")
                logger.info(f"[자막] 각 키워드의 중요도 점수: {[(k, round(keyword_importance[k], 3)) for k in transcript_keywords]}")
                
            except Exception as e:
                logger.error(f"TF-IDF 계산 중 오류: {str(e)}")
                return nx.Graph(), [], [], None
        
        if not transcript_keywords:
            logger.warning("처리할 키워드가 없습니다.")
            return nx.Graph(), [], [], None
            
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
            
            # SBERT와 SimCSE 베딩 미리 계산
            sbert = get_nlp_models()['sbert']
            simcse_model, simcse_tokenizer = get_nlp_models()['simcse']
            
            desc_sbert_embeddings = sbert.encode(desc_keywords)
            desc_inputs = simcse_tokenizer(desc_keywords, padding=True, truncation=True, return_tensors="pt")
            desc_simcse_embeddings = simcse_model(**desc_inputs).pooler_output.detach().numpy()
            
            # 각 자막 키워드에 대해 매핑 수행
            for keyword in transcript_keywords:
                # 특정 키워드 매핑 체크
                if keyword in special_mappings:
                    mapped_keyword = special_mappings[keyword]
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
            try:
                # 임베딩 일괄 계산
                embeddings = get_nlp_models()['sbert'].encode(keywords, batch_size=32)
                similarity_matrix = cosine_similarity(embeddings)
                
                # 엣지 추가 및 중심성 계산
                edge_weights = {}
                for i in range(len(keywords)):
                    for j in range(i + 1, len(keywords)):
                        similarity = similarity_matrix[i][j]
                        if similarity > EDGE_THRESHOLD:  # 엣지 임계값 상향 (0.3 → 0.5)
                            weight = similarity * (node_sizes[keywords[i]] + node_sizes[keywords[j]]) / 2
                            edge_weights[(keywords[i], keywords[j])] = weight
                            G.add_edge(keywords[i], keywords[j], weight=weight)
                
                if G.number_of_edges() > 0:  # 엣지가 있는 경우만 중심성 계산
                    centrality = nx.degree_centrality(G)
                    top_10_keywords = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                    top_10_keywords = [k[0] for k in top_10_keywords]
                else:
                    # 엣지가 없는 경우 기존 키워드에서 상위 10개 선택
                    top_10_keywords = list(keywords)[:10]
                
                # 중심 노드 설정
                central_node = top_10_keywords[0] if top_10_keywords else keywords[0]
                distances = nx.single_source_shortest_path_length(G, central_node)
                nx.set_node_attributes(G, distances, 'distance_from_center')
                
            except Exception as e:
                logger.error(f"중심성 계산 중 오류: {str(e)}")
                top_10_keywords = list(keywords)[:10]
        
        # 5. 연관어 쌍 추출
        word_pairs = []
        for u, v, data in G.edges(data=True):
            word_pairs.append(((u, v), data['weight']))
        
        # 클러스터링 수행
        if G.number_of_nodes() > 1:
            network_graph = generate_network_graph(G)
        else:
            network_graph = generate_network_graph(G)
            
        return G, sorted(word_pairs, key=lambda x: x[1], reverse=True)[:20], top_10_keywords, network_graph
        
    except Exception as e:
        logger.error(f"연관어 분석 중 오류: {str(e)}")
        return None, None, None, None

def generate_network_graph(G):
    """네트워크 그래프 시각화"""
    try:
        # 투명한 배경으로 figure 생성
        plt.figure(figsize=(16, 12), facecolor='none')
        
        # 한글 폰트 설정
        try:
            # 나눔고딕 폰트 경로 지정
# 폰트 경로 직접 지정
            if platform.system() == 'Windows':
                font_path = "C:/Windows/Fonts/malgun.ttf"
            else:  # Linux
                font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            font_prop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=font_prop.get_name())
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logger.error(f"나눔고딕 폰트 로드 실패: {e}")
            try:
                # 대체 경로 시도
                font_path = '/usr/share/fonts/nanum/NanumGothic.ttf'
                font_prop = fm.FontProperties(fname=font_path)
                plt.rc('font', family=font_prop.get_name())
                plt.rcParams['axes.unicode_minus'] = False
            except Exception as e:
                logger.error(f"대체 폰트 로드도 실패: {e}")
                # 기본 폰트 사용
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['axes.unicode_minus'] = False
        # 투명한 배경으로 figure 생성
        plt.figure(figsize=(16, 12), facecolor='none')
        
        # 한글 폰트 설정
        # try:
        #     font_path = "C:/Windows/Fonts/malgun.ttf"
        #     font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
        #     plt.rcParams['font.family'] = font_prop.get_name()
        # except:
        #     logger.warning("기본 폰트를 사용합니다.")
        #     plt.rcParams['font.family'] = 'Malgun Gothic'
        
        plt.figure(figsize=(16, 12), facecolor='white')
        # 축의 배경도 투명하게 설정
        ax = plt.gca()
        ax.set_facecolor('none')
        
        # 노드 크기와 색상 계산
        node_sizes = []
        node_colors = []
        for node in G.nodes():
            size = G.degree(node) * 500  # 연결 수에 따른 크기
            node_sizes.append(size)
            # 노드별 색상 설정
            node_colors.append('#1f77b4')  # 파란색 계열
        
        # 스프링 레이아웃으로 노드 위치 설정
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # 엣지 그리기 (가중치에 따른 색상과 두께)
        edge_weights = nx.get_edge_attributes(G, 'weight')
        if edge_weights:  # 엣지가 있는 경우에만 처리
            min_weight = min(edge_weights.values())
            max_weight = max(edge_weights.values())
            
            # 엣지를 가중치 순으로 정렬하여 그리기 (중요한 연결 위에 오도록)
            sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
            
            for u, v, data in sorted_edges:
                weight = data['weight']
                # 가중치에 따른 색상 (연한 보라색 → 진한 보라색)
                if min_weight == max_weight:
                    alpha = 0.6
                    width = 4
                else:
                    alpha = 0.3 + 0.6 * (weight - min_weight) / (max_weight - min_weight)
                    width = 1 + 8 * (weight - min_weight) / (max_weight - min_weight)
                
                nx.draw_networkx_edges(G, pos,
                                     edgelist=[(u, v)],
                                     width=width,
                                     edge_color='purple',  # 보라색으로 변경
                                     alpha=alpha,
                                     style='solid')
        
        # 노드와 레이블 그리기
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
        nx.draw_networkx_labels(G, pos, 
                              font_family=plt.rcParams['font.family'],
                              font_size=12, 
                              font_weight='bold',
                              font_color='white')  # 글씨 색상을 하얀색으로 변경
        
        # 제목 추가
        plt.title('연관어 키워드 TOP 20', 
                 fontdict={'family': plt.rcParams['font.family'], 
                          'size': 24, 
                          'weight': 'bold'}, 
                 pad=50)
        
        plt.axis('off')
        plt.tight_layout(pad=50)
        
        # 투명 배경으로 이미지 저장
        buffer = BytesIO()
        plt.savefig(buffer, format='png', 
                   bbox_inches='tight', 
                   dpi=300, 
                   facecolor='none',
                   edgecolor='none',
                   transparent=True)
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
# def get_memory_usage():
#     """현재 프로세스의 메모리 사용량을 MB 단위로 반환"""
#     import psutil
#     process = psutil.Process()
#     return process.memory_info().rss / 1024 / 1024

def get_transcript_segments(transcript):
    """자막 데이터를 시간 단위로 구분하여 텍스트로 변환"""
    transcript_segments = []
    for item in transcript:
        if 'start' in item and 'text' in item:
            start_time = int(float(item['start']))
            minutes = start_time // 60
            seconds = start_time % 60
            time_str = f"{minutes:02d}:{seconds:02d}"
            transcript_segments.append({
                'time': time_str,
                'text': item['text']
            })
    return transcript_segments

