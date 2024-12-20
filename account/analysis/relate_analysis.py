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
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertModel
# PyTorch 임포트
import torch
import re
from soynlp.normalizer import *
from django.conf import settings


# 바른 형태소 분석기 API 키 설정
API_KEY = "koba-5JNWNQI-MH5EQJY-QINVNIQ-2IOA5IY"  
# 형태소 분석기 객체 생성
t = brn.Tagger(API_KEY, "localhost")  


# 사전 학습된 모델 다운로드 및 로드를 위한 함수 정의
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
                    print(f"File magic number: {magic.hex()}")
                    
                    # 파일 포맷 확인
                    if magic.startswith(b'\xba\x16O/'):  # 특정 매직 넘버 확인
                        print("파일 포맷 확인됨")
                        
                        # 헤더 처리 개선
                        try:
                            # 헤더 라인을 바이트로 읽기
                            header_bytes = b''
                            while True:
                                byte = f.read(1)
                                if byte == b'\n':
                                    break
                                header_bytes += byte
                            
                            # 다양한 인코딩 시도
                            for encoding in ['utf-8', 'cp949', 'euc-kr']:
                                try:
                                    header = header_bytes.decode(encoding).strip()
                                    # 백으로 분리하여 숫자 추출 시
                                    parts = [part for part in header.split() if part.isdigit()]
                                    if len(parts) >= 2:
                                        vocab_size, vector_size = map(int, parts[:2])
                                        print(f"모델 정보 - 단어 수: {vocab_size}, 벡터 크기: {vector_size}")
                                        break
                                except:
                                    continue
                            else:
                                print("헤더 파싱 실패, 기본값 사용")
                                # Word2Vec 모델의 일반적인 값으로 설정
                                vocab_size, vector_size = 100000, 300
                            
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
                                    print(f"단어 처리 중 오류: {str(e)}")
                                    continue
                            
                            return model
                        except Exception as e:
                            print(f"헤더 파싱 중 오류 발생: {str(e)}")
                            return None
                    else:
                        print("알 수 없는 파일 포맷")
                        return None
                        
            except Exception as e:
                print(f"파일 읽기 실패: {str(e)}")
                return None
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return None
            
    except Exception as e:
        print(f"Word2Vec 모델 로드 중 오류 발생: {str(e)}")
        print(f"오류 타입: {type(e)}")
        return None


# 전역 변수로 Word2Vec 모델 로드 시도
try:
    print("Word2Vec 모델 로딩 시작...")
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    word2vec_model = load_pretrained_model()
    if word2vec_model is not None:
        print("Word2Vec 모델 로딩 완료")
        print(f"모델 크기: {len(word2vec_model.key_to_index)} 단어")
    else:
        print("Word2Vec 모델 로딩 실패")
except Exception as e:
    print(f"모델 로드 중 예외 발생: {str(e)}")
    word2vec_model = None


# 한국어SBERT와 KoSimCSE 모델을 로드하는 함수 정의
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


# KPF-BERT 모델 로드 함수 수정
def load_kpf_bert():
    """한국어 BERT 모델과 토크나이저를 로드하는 함수"""
    try:
        # KLUE-BERT 모델 사용 (공개 모델)
        model_name_or_path = "klue/bert-base"
        
        # 모델과 토크나이저 로드
        tokenizer = BertTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=False  # 한국어는 소문자화하지 않음
        )
        
        model = BertModel.from_pretrained(
            model_name_or_path,
            add_pooling_layer=False
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        return tokenizer, model
        
    except Exception as e:
        print(f"BERT 모델 로드 실패: {str(e)}")
        # 대체 모델 시도
        try:
            # KoCharELECTRA 모델 시도 (또 다른 공개 모델)
            model_name_or_path = "monologg/kocharelectra-base-discriminator"
            
            tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
            model = BertModel.from_pretrained(model_name_or_path)
            
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            
            return tokenizer, model
            
        except Exception as fallback_e:
            print(f"대체 모델 로드도 실패: {str(fallback_e)}")
            return None, None

# 전역 변수로 KPF-BERT 모델 로드
kpf_tokenizer, kpf_model = load_kpf_bert()

def correct_text_with_bert(text, window_size=5):
    """
    KPF-BERT를 사용한 문맥 기반 오타 수정 함수
    """
    try:
        if kpf_tokenizer is None or kpf_model is None:
            return text

        # 토큰화
        tokens = kpf_tokenizer.tokenize(text)
        if not tokens:
            return text
        
        corrected_tokens = []
        
        # 슬라이딩 윈도우로 문맥 고려
        for i in range(len(tokens)):
            current_token = tokens[i]
            
            # 특수 토큰이나 숫자는 건너뛰기
            if current_token.startswith('##') or current_token.isdigit():
                corrected_tokens.append(current_token)
                continue
            
            # 문맥 윈도우 생성
            start_idx = max(0, i - window_size)
            end_idx = min(len(tokens), i + window_size + 1)
            context = tokens[start_idx:i] + [kpf_tokenizer.mask_token] + tokens[i+1:end_idx]
            
            # BERT 입력 생성
            inputs = kpf_tokenizer.encode_plus(
                ' '.join(context),
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # BERT로 예측
            with torch.no_grad():
                outputs = kpf_model(**inputs)
                predictions = outputs.logits
            
            # MASK 토큰 위치의 예측값 추출
            mask_idx = torch.where(inputs['input_ids'] == kpf_tokenizer.mask_token_id)[1]
            if len(mask_idx) == 0:
                corrected_tokens.append(current_token)
                continue
                
            mask_predictions = predictions[0, mask_idx[0]]
            
            # 상위 5개 예측 중에서 원본 토큰과 가장 유사한 것 선택
            top_5 = torch.topk(mask_predictions, 5)
            candidates = [kpf_tokenizer.convert_ids_to_tokens(idx.item()) 
                        for idx in top_5.indices]
            
            # 원본 토큰과 가장 유사한 후보 선택
            best_candidate = current_token
            max_similarity = 0
            
            for candidate in candidates:
                # 자카드 유사도 계산
                similarity = len(set(current_token) & set(candidate)) / \
                           len(set(current_token) | set(candidate))
                if similarity > max_similarity and similarity > 0.5:  # 임계값
                    max_similarity = similarity
                    best_candidate = candidate
            
            corrected_tokens.append(best_candidate)
        
        # 토큰을 다시 문장으로 조합
        return kpf_tokenizer.convert_tokens_to_string(corrected_tokens)
        
    except Exception as e:
        print(f"텍스트 교정 중 오류 발생: {str(e)}")
        return text


# 유튜브 영상 설명에서 연관 단어를 분석하여 네트워크 그래프를 생성하는 함수
def analyze_related_words(video_desc):
    """
    유튜브 영상 설명에서 연관 단어를 분석하여 네트워크 그래프를 생성하는 함수
    """
    try:
        # 텍스트 전처리 및 오타 수정
        if kpf_model is not None:
            video_desc = correct_text_with_bert(video_desc)
            print("오타 수정된 텍스트:", video_desc)
        
        # 불용어 처리: 분석에서 제외할 단어들을 파일에서 로드
        stopwords = set()
        try:
            with open('account/static/불용어.txt', 'r', encoding='utf-8') as f:
                stopwords = set(f.read().splitlines())
        except Exception as e:
            print(f"불용어 파일 로드 중 오류: {str(e)}")
        
        # 입력 텍스트가 바이트 문자열인 경우 반 문자열로 변환
        if isinstance(video_desc, bytes):
            video_desc = video_desc.decode('utf-8')  
        
        # 입력된 비디오 설명 출력
        print("입력된 비디오 설명:", video_desc)
        
        # 빈 텍스트인 경우 빈 그래프 반환
        if not video_desc:
            print("비디오 설명이 없습니다!")
            return nx.Graph(), [], []
            
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
            
            # 문맥 윈도 생성
            context_windows = []
            window_size = 5
            for i in range(len(words)):
                start = max(0, i - window_size)
                end = min(len(words), i + window_size)
                context = ' '.join(words[start:end])
                context_windows.append(context)
            
            # NLP 모델들이 로드된 경우 고급 분석 수행
            if nlp_models is not None and word2vec_model is not None:
                print("NLP 모델을 사용한 분석 시작...")
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
                            # 단어가 모델에 있는지 확인
                            if word1 in word2vec_model and word2 in word2vec_model:
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
                                    
                                    print(f"단어 쌍 추가: {word1} - {word2} (점수: {final_score:.3f})")
                            else:
                                # 기본 분석으로 대체
                                if j < i + 5:  # 문맥 윈도우 크기 내에서만
                                    word_pair = tuple(sorted([word1, word2]))
                                    word_pairs.append((word_pair, 0.5))  # 기본 점수 부여
                                    
                                    # 단어별 중요도 누적
                                    word_importance[word1] = word_importance.get(word1, 0) + 0.5
                                    word_importance[word2] = word_importance.get(word2, 0) + 0.5
                                    
                        except KeyError as e:
                            print(f"단어 처리 중 KeyError: {str(e)}")
                            continue
                        except Exception as e:
                            print(f"단어 쌍 처리 중 오류: {str(e)}")
                            continue
            else:
                print("기본 분석 방식 사용...")
                # 기본 분석 수행
                for i in range(len(words)-1):
                    for j in range(i+1, min(i+5, len(words))):
                        word1, word2 = words[i], words[j]
                        word_pair = tuple(sorted([word1, word2]))
                        word_pairs.append((word_pair, 1))
                        
                        # 단어별 중요도 누적
                        word_importance[word1] = word_importance.get(word1, 0) + 1
                        word_importance[word2] = word_importance.get(word2, 0) + 1
            
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
                
                # 그래프가 비어있지 않은 경우에만 중심성 계산
                if len(G.nodes()) > 0:
                    try:
                        centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
                    except:
                        # 중심성 계산 실패시 degree centrality 사용
                        centrality = nx.degree_centrality(G)
                    
                    # 중요 노드 선택
                    important_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
                    important_words = set(word for word, _ in important_nodes)
                    
                    # 중요 노드로 서브그래프 생성
                    G = G.subgraph(important_words).copy()
                else:
                    print("그래프에 노드가 없습니다.")
                    return nx.Graph(), [], []
            
            # 위 관계 추출
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
    try:
        # matplotlib 백엔드 설정
        matplotlib.use('Agg')
        
        # 한글 폰트 설정
        font_path = os.path.join(settings.STATIC_ROOT, 'fonts', 'NanumGothic.ttf')
        if not os.path.exists(font_path):
            # STATIC_ROOT에 없으면 STATICFILES_DIRS에서 찾기
            for static_dir in settings.STATICFILES_DIRS:
                alt_font_path = os.path.join(static_dir, 'fonts', 'NanumGothic.ttf')
                if os.path.exists(alt_font_path):
                    font_path = alt_font_path
                    break
        
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['font.size'] = 12

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