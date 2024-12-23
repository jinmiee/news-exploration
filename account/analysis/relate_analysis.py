from collections import Counter, defaultdict
import networkx as nx
import bareunpy as brn
import unicodedata
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
from time import time
import pandas as pd
from .clustering import find_optimal_k, evaluate_model_performance
from .visualization import generate_network_graph, visualize_performance_metrics




# 로깅 설정
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 추가 로깅 설정
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('gensim').setLevel(logging.ERROR)

# 상수 정의
API_KEY = "koba-5JNWNQI-MH5EQJY-QINVNIQ-2IOA5IY"
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

# 성능 메트릭을 저장할 전역 변수
performance_metrics = defaultdict(list)

# def log_performance_metrics(metrics):
#     """성능 지표를 기록하는 함수"""
#     for key, value in metrics.items():
#         performance_metrics[key].append(value)
    
#     # 주기적으로 CSV 파일로 저장
#     if len(performance_metrics['timestamp']) % 100 == 0:  # 100 단위로 저장
#         pd.DataFrame(performance_metrics).to_csv('analysis_performance_metrics.csv', index=False)

def load_sbert_model():
    """
    한국어 SBERT와 KoSimCSE 모델을 로드하는 함수
    """
    try:
        # 두 가지 모델을 딕셔너리 형태로 저장
        models = {
            # SBERT 모델 로드 - 한국어 문장 임베딩을 위한 모델
            # CPU에서 실행되도록 device='cpu' 설정
            'sbert': SentenceTransformer('jhgan/ko-sbert-nli', device='cpu'),
            
            # KoSimCSE 모델 로드 - 문장 유사도 계산을 위한 모델
            # 모델과 토크나이저를 튜플 형태로 저장
            'simcse': (
                # 사전학습된 RoBERTa 기반 모델 로드
                # revision='main'은 최신 버전 사용
                # use_auth_token=False로 인증 없이 사용
                AutoModel.from_pretrained(
                    'BM-K/KoSimCSE-roberta',
                    revision='main',
                    token=None
                ),
                # 해당 모델의 토크나이저 로드
                AutoTokenizer.from_pretrained(
                    'BM-K/KoSimCSE-roberta', 
                    revision='main',
                    token=None
                )
            )
        }
        logger.info("SBERT와 KoSimCSE 모델 로드 완료")
        return models
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")
        return None

# NLP 모델 로드
try:
    logger.info("NLP 모델 로딩 시작...")
    nlp_models = load_sbert_model()
    if nlp_models is not None:
        logger.info("NLP 모델 로딩 완료")
    else:
        logger.warning("NLP 모델 로딩 실패")
except Exception as e:
    logger.error(f"모델 로드 중 예외 발생: {str(e)}")
    nlp_models = None

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
                            not keyword.isdigit() and  # 숫자만으로 구성되지 않음
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

def analyze_related_words(text, transcript, clean_title_func=None):
    """
    텍스트와 자막에서 연관어를 분석하는 함수
    """
    try:
        print("\n[연관어 분석 시작]")
        
        # clean_title_func가 전달되지 않았다면 원본 텍스트 사용
        if clean_title_func:
            cleaned_desc = clean_title_func(text)
        else:
            cleaned_desc = text

        # 특정 키워드 매핑 정의
        special_mappings = {
            "윤성": "윤석열",
            "윤성열": "윤석열"
        }
        
        # 1. 설명에서 모든 키워드 추출
        desc_keywords = extract_keywords_from_desc(cleaned_desc)
        print(f"[설명 키워드] 추출된 키워드 ({len(desc_keywords)}개): {desc_keywords}")
        
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
            print(f"[자막] 전체 추출 키워드 수: {len(all_transcript_keywords)}개")
            print(f"[자막] 추출된 전체 키워드: {all_transcript_keywords}")  # 디버깅용
            
            if not all_transcript_keywords:
                print("[경고] 자막에서 추출된 키워드가 없습니다")
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
                    # if any(topic in keyword for topic in ['탄핵', '대통령', '윤석열', '민주당', '국회']):
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
                
                print(f"[자막] 선정된 상위 {TOP_KEYWORDS_COUNT}개 키워드: {transcript_keywords}")
                print(f"[자막] 각 키워드의 중요도 점수: {[(k, round(keyword_importance[k], 3)) for k in transcript_keywords]}")
                
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
                embeddings = nlp_models['sbert'].encode(keywords)
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
        
        # 성능 평가 그래프 생성
        if G.number_of_nodes() > 1:
            node_embeddings = nlp_models['sbert'].encode(list(G.nodes()))
            
            # 최적의 클러스터 수 찾기 및 차원 축소
            optimal_k, reduced_embeddings, cluster_labels, sil_score = find_optimal_k(node_embeddings)
            
            # 클러스터링 결과 시각화
            performance_graph = evaluate_model_performance(
                reduced_embeddings, 
                cluster_labels,
                sil_score
            )
            
            # 네트워크 그래프에 클러스터 정보 반영
            network_graph = generate_network_graph(G, cluster_labels)
            
        else:
            performance_graph = None
            network_graph = generate_network_graph(G)
        
        return G, sorted(word_pairs, key=lambda x: x[1], reverse=True)[:20], top_10_keywords, performance_graph
        
    except Exception as e:
        logger.error(f"연관어 분석 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return nx.Graph(), [], [], None

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
            graph, top_pairs, important_keywords, performance_graph = analyze_related_words(
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
            
            # 성능 메트릭 시각화 추가
            performance_graph = visualize_performance_metrics(performance_metrics)
            
            context = {
                'section': 'relate',
                'video': video,
                'video_title': cleaned_title,
                'network_graph': network_graph,
                'top_pairs': top_pairs,
                'categorized_news': categorized_news,
                'important_keywords': important_keywords,
                'transcript_segments': transcript_segments,
                'performance_graph': performance_graph
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
