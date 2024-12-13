from collections import Counter  # 단어 쌍의 빈도수를 계산하기 위한 라이브러리
import networkx as nx  # 네트워크 그래프를 생성하고 조작하기 위한 라이브러리
import matplotlib  # 그래프 시각화를 위한 라이브러리
matplotlib.use('Agg')  # 서버 환경에서 그래프를 생성하기 위해 백엔드 설정
import matplotlib.pyplot as plt  # 그래프 그리기 기능
import matplotlib.font_manager  # 한글 폰트 관리
from io import BytesIO  # 메모리 상에서 바이트 데이터를 다루기 위한 클래스
import base64  # 바이너리 데이터를 텍스트로 인코딩하기 위한 라이브러리
import bareunpy as brn  # 한국어 형태소 분석을 위한 바른 형태소 분석기
import unicodedata  # 유니코드 문자 처리를 위한 라이브러리
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import urllib.request
import os
from collections import defaultdict
import math
from sentence_transformers import SentenceTransformer
import torch
import pickle
import os.path


# 바른 형태소 분석기 초기화
API_KEY = "koba-5JNWNQI-MH5EQJY-QINVNIQ-2IOA5IY"  # 바른 형태소 분석기 API 키
t = brn.Tagger(API_KEY, "localhost")  # 형태소 분석기 객체 생성


# 사전 학습된 모델 다운로드 및 로드를 위한 함수 추가
def load_pretrained_model():
    """
    사전 학습된 한국어 Word2Vec 모델을 로드하는 함수
    피클 파일이 있으면 피클에서 로드하고, 없으면 다운로드 후 피클로 저장
    """
    # 디렉토리 경로 설정
    static_dir = 'account/static'
    model_pickle_path = f'{static_dir}/word2vec_model.pkl'
    model_bin_path = f'{static_dir}/ko.bin'
    
    # static 디렉토리가 없으면 생성
    if not os.path.exists(static_dir):
        try:
            os.makedirs(static_dir)
            print(f"디렉토리 생성됨: {static_dir}")
        except Exception as e:
            print(f"디렉토리 생성 실패: {str(e)}")
            return None
    
    # 피클 파일이 존재하면 피클에서 로드
    if os.path.exists(model_pickle_path):
        print("피클 파일에서 Word2Vec 모델 로드 중...")
        try:
            with open(model_pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"피클 파일 로드 실패: {str(e)}")
            # 피클 파일이 손상된 경우 삭제
            os.remove(model_pickle_path)
    
    # 피클 파일이 없으면 bin 파일에서 로드 후 피클로 저장
    print("Word2Vec 모델 새로 로드 중... 시간이 다소 소요됩니다.")
    try:
        if not os.path.exists(model_bin_path):
            print("사전 학습된 모델 다운로드 중...")
            urllib.request.urlretrieve(
                'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz',
                model_bin_path
            )
        
        # bin 파일에서 모델 로드
        model = KeyedVectors.load_word2vec_format(model_bin_path, binary=True)
        
        # 피클로 저장
        print("모델을 피클 파일로 저장 중...")
        with open(model_pickle_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        return None

# 전역 변수로 모델 로드
try:
    word2vec_model = load_pretrained_model()
except Exception as e:
    print(f"모델 로드 중 오류 발생: {str(e)}")
    word2vec_model = None


def load_sbert_model():
    """
    한국어 Sentence-BERT 모델을 로드하는 함수
    """
    try:
        model_name = 'jhgan/ko-sbert-nli'
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"SBERT 모델 로드 중 오류 발생: {str(e)}")
        return None

# 전역 변수로 SBERT 모델 로드 (word2vec_model 선언 아래에 추가)
try:
    sbert_model = load_sbert_model()
except Exception as e:
    print(f"SBERT 모델 로드 중 오류 발생: {str(e)}")
    sbert_model = None


def analyze_related_words(video_desc):
    """
    유튜브 영상 설명에서 연관 단어를 분석하여 네트워크 그래프를 생성하는 함수
    
    Args:
        video_desc (str): 분석할 유튜브 영상의 설명 텍스트
    
    Returns:
        tuple: (networkx.Graph 객체, 연관 단어 쌍 리스트)
    """
    try:
        # 1. 불용어 처리: 분석에서 제외할 단어들을 파일에서 로드
        stopwords = set()
        try:
            with open('account/static/불용어.txt', 'r', encoding='utf-8') as f:
                stopwords = set(f.read().splitlines())
        except Exception as e:
            print(f"불용어 파일 로드 중 오류: {str(e)}")
        
        # 2. 입력 텍스트 전처리
        if isinstance(video_desc, bytes):
            video_desc = video_desc.decode('utf-8')  # 바이트 문자열을 일반 문자열로 변환
        
        print("입력된 비디오 설명:", video_desc)
        
        # 3. 빈 텍스트 체크
        if not video_desc:
            print("비디오 설명이 없습니다!")
            return nx.Graph(), []
            
        # 4. 형태소 분석 수행
        print("형태소 분석 시작...")
        tagged = t.tags([video_desc])
        
        # 5. 명사 추출
        nouns = []
        for sent in tagged.sentences():  # 문장별로 처리
            for token in sent.tokens:    # 각 토큰(단어)별로 처리
                for morph in token.morphemes:  # 각 형태소별로 처리
                    if morph.tag == 24 or morph.tag == 25:  # 24:일반명사, 25:고유명사
                        try:
                            # 형태소의 텍스트 추출 및 처리
                            noun = morph.text.content
                            if isinstance(noun, bytes):
                                noun = noun.decode('utf-8')
                            
                            # 불용어가 아닌 경우에만 명사 리스트에 추가
                            if noun not in stopwords:
                                print(f"추출된 명사: {noun}")
                                nouns.append(noun)
                        except Exception as e:
                            print(f"명사 처리 중 오류 발생: {str(e)}")
                            continue
        
        # 6. 2글자 이상의 명사만 선택 (의미있는 단어만 선택)
        words = [word for word in nouns if len(str(word)) > 1]
        print("필터링된 단어 목록:", words)
        
        # 7. 연관 단어 분석 개선
        if len(words) > 1:
            # TF-IDF 가중치 계산을 위한 단어 빈도수
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            
            word_pairs = []
            word_importance = {}  # 단어별 중요도 저장
            
            # 문맥 윈도우 내의 단어들을 문장으로 결합
            context_windows = []
            window_size = 5
            for i in range(len(words)):
                start = max(0, i - window_size)
                end = min(len(words), i + window_size)
                context = ' '.join(words[start:end])
                context_windows.append(context)
            
            if sbert_model is not None and word2vec_model is not None:
                # SBERT 임베딩 계산
                context_embeddings = sbert_model.encode(context_windows)
                
                for i, word1 in enumerate(words):
                    for j, word2 in enumerate(words[i+1:], i+1):
                        try:
                            # 1. Word2Vec 유사도
                            w2v_similarity = word2vec_model.similarity(word1, word2)
                            
                            # 2. SBERT 문맥 유사도
                            context1_embed = context_embeddings[i]
                            context2_embed = context_embeddings[j]
                            sbert_similarity = cosine_similarity(
                                context1_embed.reshape(1, -1), 
                                context2_embed.reshape(1, -1)
                            )[0][0]
                            
                            # 3. 거리 기반 가중치
                            distance_weight = 1.0 / (j - i)
                            
                            # 4. 빈도 기반 가중치
                            freq_weight = (word_freq[word1] + word_freq[word2]) / len(words)
                            
                            # 5. 최종 유사도 점수 계산 (가중치 조정)
                            final_score = (
                                0.4 * w2v_similarity + 
                                0.3 * sbert_similarity + 
                                0.2 * distance_weight + 
                                0.1 * freq_weight
                            )
                            
                            if final_score > 0.3:  # 임계값
                                word_pair = tuple(sorted([word1, word2]))
                                word_pairs.append((word_pair, final_score))
                                
                                # 단어별 중요도 누적
                                word_importance[word1] = word_importance.get(word1, 0) + final_score
                                word_importance[word2] = word_importance.get(word2, 0) + final_score
                                
                        except KeyError:
                            # Word2Vec에 없는 단어는 SBERT만 사용
                            try:
                                sbert_similarity = cosine_similarity(
                                    context1_embed.reshape(1, -1), 
                                    context2_embed.reshape(1, -1)
                                )[0][0]
                                
                                if sbert_similarity > 0.3:
                                    word_pair = tuple(sorted([word1, word2]))
                                    word_pairs.append((word_pair, sbert_similarity))
                            except:
                                continue
            else:
                # 기존 방식으로 단어 쌍 생성
                for i in range(len(words)-1):
                    for j in range(i+1, min(i+5, len(words))):
                        word1, word2 = words[i], words[j]
                        word_pairs.append((tuple(sorted([word1, word2])), 1))
            
            # 빈도수 계산 (기존 방식 유지)
            pair_counts = Counter(word[0] for word in word_pairs)
            
            # 네트워크 그래프 생성
            G = nx.Graph()
            
            if word2vec_model is not None:
                # 중요도 기반 노드 크기 설정을 위해 속성 추가
                for word, importance in word_importance.items():
                    G.add_node(word, importance=importance)
                
                for (word1, word2), score in word_pairs:
                    if G.has_edge(word1, word2):
                        G[word1][word2]['weight'] = max(G[word1][word2]['weight'], score)
                    else:
                        G.add_edge(word1, word2, weight=score)
                
                # 연결 중심성 계산
                centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
                
                # 중심성이 높은 노드 중심으로 서브그래프 추출
                important_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
                important_words = set(word for word, _ in important_nodes)
                
                # 중요 노드들로 구성된 서브그래프 생성
                G = G.subgraph(important_words).copy()
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



def generate_network_graph(G):
    """네트워크 그래프를 생성하는 함수"""
    if not G.nodes():
        # 빈 그래프일 경우 빈 이미지 반환
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, '연관어를 찾을 수 없습니다.', 
                horizontalalignment='center',
                verticalalignment='center')
        plt.axis('off')
        
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
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 환경
    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rc('font', family=font_name)
    
    # 그래프 크기 조절
    plt.figure(figsize=(8, 6))
    
    # 노드 크기 설정
    degrees = dict(G.degree())
    node_size = [v * 100 for v in degrees.values()]
    
    # 엣지 굵기 설정
    edge_width = [G[u][v].get('weight', 1.0) * 0.3 for u, v in G.edges()]
    
    # 그래프 레이아웃 설정
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    
    # 그래프 그리기
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.4)
    
    # 한글 레이블 설정
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_family=font_name, font_size=8)
    
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

# 중요 키워드 추출 로직 추가
def get_important_keywords(G, top_n=5):
    """중요 키워드를 추출하는 함수"""
    if not G.nodes():
        return []
    
    try:
        # 연결 요소가 여러 개인 경우를 처리
        if nx.number_connected_components(G) > 1:
            # 가장 큰 연결 요소만 선택
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # degree centrality 사용 (더 안정적인 방법)
        centrality = nx.degree_centrality(G)
        important_keywords = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, _ in important_keywords]
    except Exception as e:
        print(f"중요 키워드 추출 중 오류 발생: {str(e)}")
        # 실패 �� degree 기반으로 간단히 처리
        degrees = dict(G.degree())
        sorted_words = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, _ in sorted_words]