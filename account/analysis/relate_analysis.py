# 단어 쌍의 빈도수를 계산하기 위한 Counter 라이브러리 임포트
from collections import Counter  
# 네트워크 그래프를 생성하고 조작하기 위한 networkx 라이브러리 임포트
import networkx as nx  
# 그래프 시각화를 위한 matplotlib 라이브러리 임포트
import matplotlib  
# 서버 환경에서 그래프를 생성하기 위해 백엔드 설정
matplotlib.use('Agg')  
# matplotlib의 pyplot 모듈 임포트
import matplotlib.pyplot as plt  
# matplotlib의 한글 폰트 관리 모듈 임포트
import matplotlib.font_manager  
# 메모리 상에서 바이트 데이터를 다루기 위한 BytesIO 클래스 임포트
from io import BytesIO  
# 바이너리 데이터를 텍스트로 인코딩하기 위한 base64 라이브러리 임포트
import base64  
# 한국어 형태소 분석을 위한 바른 형태소 분석기 임포트
import bareunpy as brn  
# 유니코드 문자 처리를 위한 라이브러리 임포트
import unicodedata  
# Word2Vec 모델 임포트
from gensim.models import Word2Vec
# 수치 연산을 위한 numpy 임포트
import numpy as np
# 코사인 유사도 계산을 위한 함수 임포트
from sklearn.metrics.pairwise import cosine_similarity
# 사전학습된 Word2Vec 모델을 로드하기 위한 KeyedVectors 임포트
from gensim.models import KeyedVectors
# URL 처리를 위한 라이브러리 임포트
import urllib.request
# 파일 시스템 관련 기능을 위한 os 모듈 임포트
import os
# 기본값이 있는 딕셔너리를 위한 defaultdict 임포트
from collections import defaultdict
# 수학 연산을 위한 math 모듈 임포트
import math
# 문장 임베딩을 위한 SentenceTransformer 임포트
from sentence_transformers import SentenceTransformer
# 트랜스포머 모델과 토크나이저 임포트
from transformers import AutoModel, AutoTokenizer
# PyTorch 임포트
import torch


# 바른 형태소 분석기 API 키 설정
API_KEY = "koba-5JNWNQI-MH5EQJY-QINVNIQ-2IOA5IY"  
# 형태소 분석기 객체 생성
t = brn.Tagger(API_KEY, "localhost")  


# 사전 학습된 모델 다운로드 및 로드를 위한 함수 정의
def load_pretrained_model():
    """
    사전 학습된 한국어 Word2Vec 모델을 로드하는 함수
    """
    # 모델 파일 경로 설정
    model_path = 'account/static/ko.bin'
    # 모델 파일이 존재하는 경우 로드
    if os.path.exists(model_path):
        return KeyedVectors.load_word2vec_format(model_path, binary=True)
    # 모델 파일이 없는 경우 메시지 출력
    else:
        print("모델 파일을 찾을 수 없습니다: account/static/ko.bin")
        return None


# 전역 변수로 Word2Vec 모델 로드 시도
try:
    word2vec_model = load_pretrained_model()
# 모델 로드 실패 시 예외 처리
except Exception as e:
    print(f"모델 로드 중 오류 발생: {str(e)}")
    word2vec_model = None


# SBERT와 KoSimCSE 모델을 로드하는 함수 정의
def load_sbert_model():
    """
    한국어 SBERT와 KoSimCSE 모델을 로드하는 함수
    """
    try:
        # 두 가지 모델을 딕셔너리 형태로 로드
        models = {
            'sbert': SentenceTransformer('jhgan/ko-sbert-nli'),
            'simcse': (
                AutoModel.from_pretrained('BM-K/KoSimCSE-roberta'),
                AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
            )
        }
        return models
    # 모델 로드 실패 시 예외 처리
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        return None

# 전역 변수로 NLP 모델들 로드 시도
try:
    nlp_models = load_sbert_model()
# 모델 로드 실패 시 예외 처리
except Exception as e:
    print(f"모델 로드 중 오류 발생: {str(e)}")
    nlp_models = None




# 유튜브 영상 설명에서 연관 단어를 분석하여 네트워크 그래프를 생성하는 함수
def analyze_related_words(video_desc):
    """
    유튜브 영상 설명에서 연관 단어를 분석하여 네트워크 그래프를 생성하는 함수
    
    Args:
        video_desc (str): 분석할 유튜브 영상의 설명 텍스트
    
    Returns:
        tuple: (networkx.Graph 객체, 연관 단어 쌍 리스트)
    """
    try:
        # 불용어 처리: 분석에서 제외할 단어들을 파일에서 로드
        stopwords = set()
        try:
            with open('account/static/불용어.txt', 'r', encoding='utf-8') as f:
                stopwords = set(f.read().splitlines())
        except Exception as e:
            print(f"불용어 파일 로드 중 오류: {str(e)}")
        
        # 입력 텍스트가 바이트 문자열인 경우 일반 문자열로 변환
        if isinstance(video_desc, bytes):
            video_desc = video_desc.decode('utf-8')  
        
        # 입력된 비디오 설명 출력
        print("입력된 비디오 설명:", video_desc)
        
        # 빈 텍스트인 경우 빈 그래프 반환
        if not video_desc:
            print("비디오 설명이 없습니다!")
            return nx.Graph(), []
            
        # 형태소 분석 수행
        print("형태소 분석 시작...")
        tagged = t.tags([video_desc])
        
        # 명사 추출
        nouns = []
        # 문장별로 처리
        for sent in tagged.sentences():  
            # 각 토큰(단어)별로 처리
            for token in sent.tokens:    
                # 각 형태소별로 처리
                for morph in token.morphemes:  
                    # 일반명사(24)나 고유명사(25)인 경우만 처리
                    if morph.tag == 24 or morph.tag == 25:  
                        try:
                            # 형태소의 텍스트 추출
                            noun = morph.text.content
                            # 바이트 문자열인 경우 변환
                            if isinstance(noun, bytes):
                                noun = noun.decode('utf-8')
                            
                            # 불용어가 아닌 경우에만 명사 리스트에 추가
                            if noun not in stopwords:
                                print(f"추출된 명사: {noun}")
                                nouns.append(noun)
                        except Exception as e:
                            print(f"명사 처리 중 오류 발생: {str(e)}")
                            continue
        
        # 2글자 이상의 명사만 선택 (의미있는 단어만 선택)
        words = [word for word in nouns if len(str(word)) > 1]
        print("필터링된 단어 목록:", words)
        
        # 연관 단어 분석 수행
        if len(words) > 1:
            # TF-IDF 가중치 계산을 위한 단어 빈도수 계산
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            
            # 단어 쌍과 중요도를 저장할 리스트와 딕셔너리 초기화
            word_pairs = []
            word_importance = {}  
            
            # 문맥 윈도우 생성
            context_windows = []
            window_size = 5
            for i in range(len(words)):
                start = max(0, i - window_size)
                end = min(len(words), i + window_size)
                context = ' '.join(words[start:end])
                context_windows.append(context)
            
            # NLP 모델들이 로드된 경우 고급 분석 수행
            if nlp_models is not None and word2vec_model is not None:
                # SBERT 임베딩 계산
                sbert_embeddings = nlp_models['sbert'].encode(context_windows)
                
                # SimCSE 임베딩 계산
                simcse_model, simcse_tokenizer = nlp_models['simcse']
                simcse_inputs = simcse_tokenizer(context_windows, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    simcse_embeddings = simcse_model(**simcse_inputs).last_hidden_state[:, 0, :].numpy()
                
                # 각 단어 쌍에 대해 유사도 계산
                for i, word1 in enumerate(words):
                    for j, word2 in enumerate(words[i+1:], i+1):
                        try:
                            # Word2Vec 유사도 계산
                            w2v_similarity = word2vec_model.similarity(word1, word2)
                            
                            # SBERT 문맥 유사도 계산
                            sbert_similarity = cosine_similarity(
                                sbert_embeddings[i].reshape(1, -1),
                                sbert_embeddings[j].reshape(1, -1)
                            )[0][0]
                            
                            # SimCSE 문맥 유사도 계산
                            simcse_similarity = cosine_similarity(
                                simcse_embeddings[i].reshape(1, -1),
                                simcse_embeddings[j].reshape(1, -1)
                            )[0][0]
                            
                            # 거리 기반 가중치 계산
                            distance_weight = 1.0 / (j - i)
                            
                            # 빈도 기반 가중치 계산
                            freq_weight = (word_freq[word1] + word_freq[word2]) / len(words)
                            
                            # 최종 유사도 점수 계산
                            final_score = (
                                0.3 * w2v_similarity + 
                                0.25 * sbert_similarity +
                                0.25 * simcse_similarity +
                                0.1 * distance_weight + 
                                0.1 * freq_weight
                            )
                            
                            # 임계값을 넘는 경우만 저장
                            if final_score > 0.3:  
                                word_pair = tuple(sorted([word1, word2]))
                                word_pairs.append((word_pair, final_score))
                                
                                # 단어별 중요도 누적
                                word_importance[word1] = word_importance.get(word1, 0) + final_score
                                word_importance[word2] = word_importance.get(word2, 0) + final_score
                                
                        except KeyError:
                            continue
            # NLP 모델이 없는 경우 기본 분석 수행
            else:
                for i in range(len(words)-1):
                    for j in range(i+1, min(i+5, len(words))):
                        word1, word2 = words[i], words[j]
                        word_pairs.append((tuple(sorted([word1, word2])), 1))
            
            # 단어 쌍의 빈도수 계산
            pair_counts = Counter(word[0] for word in word_pairs)
            
            # 네트워크 그래프 생성
            G = nx.Graph()
            
            # Word2Vec 모델이 있는 경우 고급 그래프 생성
            if word2vec_model is not None:
                # 중요도 기반으로 노드 속성 추가
                for word, importance in word_importance.items():
                    G.add_node(word, importance=importance)
                
                # 엣지 추가
                for (word1, word2), score in word_pairs:
                    if G.has_edge(word1, word2):
                        G[word1][word2]['weight'] = max(G[word1][word2]['weight'], score)
                    else:
                        G.add_edge(word1, word2, weight=score)
                
                # 연결 중심성 계산
                centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
                
                # 중요 노드 선택
                important_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
                important_words = set(word for word, _ in important_nodes)
                
                # 중요 노드로 서브그래프 생성
                G = G.subgraph(important_words).copy()
            # Word2Vec 모델이 없는 경우 기본 그래프 생성
            else:
                for (word1, word2), count in pair_counts.most_common(30):
                    G.add_edge(word1, word2, weight=count)
            
            # 상위 관계 추출
            if word2vec_model is not None:
                filtered_pairs = [(pair, score) for pair, score in word_pairs 
                                if pair[0] in G.nodes and pair[1] in G.nodes]
                top_pairs = [(pair[0], pair[1]) for pair in filtered_pairs[:20]]
            else:
                top_pairs = [(pair[0], pair[1]) for pair in pair_counts.most_common(20)]
            
            # 중요 키워드 추출
            important_keywords = get_important_keywords(G)
            
            return G, top_pairs, important_keywords
            
        return nx.Graph(), [], []
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return nx.Graph(), [], []


# 네트워크 그래프를 시각화하는 함수
def generate_network_graph(G):
    """네트워크 그래프를 생성하는 함수"""
    # 빈 그래프인 경우 처리
    if not G.nodes():
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, '연관어를 찾을 수 없습니다.', 
                horizontalalignment='center',
                verticalalignment='center')
        plt.axis('off')
        
        # 이미지를 바이트로 변환
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=200)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    # matplotlib 백엔드 설정
    matplotlib.use('Agg')
    
    # 한글 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  
    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rc('font', family=font_name)
    
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
    nx.draw_networkx_labels(G, pos, labels, font_family=font_name, font_size=8)
    
    # 그래프 제목 설정
    plt.title('연관어 네트워크', fontdict={'family': font_name, 'size': 12}, pad=20)
    plt.axis('off')
    
    # 그래프를 이미지로 변환
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=200, facecolor='white')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

# 중요 키워드를 추출하는 함수
def get_important_keywords(G, top_n=5):
    """중요 키워드를 추출하는 함수"""
    # 빈 그래프인 경우 처리
    if not G.nodes():
        return []
    
    try:
        # 연결 요소가 여러 개인 경우 처리
        if nx.number_connected_components(G) > 1:
            # 가장 큰 연결 요소 선택
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # 중심성 계산 및 중요 키워드 추출
        centrality = nx.degree_centrality(G)
        important_keywords = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, _ in important_keywords]
    except Exception as e:
        print(f"중요 키워드 추출 중 오류 발생: {str(e)}")
        # 실패 시 단순 degree 기반으로 처리
        degrees = dict(G.degree())
        sorted_words = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, _ in sorted_words]