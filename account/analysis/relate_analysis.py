from konlpy.tag import Okt
from collections import Counter
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager
from io import BytesIO
import base64
import matplotlib.pyplot as plt

def analyze_related_words(text):
    okt = Okt()
    
    # 명사 추출
    nouns = okt.nouns(text)
    
    # 2글자 이상의 명사만 선택
    words = [word for word in nouns if len(word) > 1]
    
    # 단어 쌍 생성
    word_pairs = []
    for i in range(len(words)-1):
        for j in range(i+1, min(i+5, len(words))):
            word_pairs.append(tuple(sorted([words[i], words[j]])))
    
    # 단어 쌍 빈도수 계산
    pair_counts = Counter(word_pairs)
    
    # 네트워크 그래프 생성
    G = nx.Graph()
    
    # 상위 30개의 연관 관계만 사용
    for (word1, word2), count in pair_counts.most_common(30):
        G.add_edge(word1, word2, weight=count)
    
    return G, pair_counts.most_common(30)

def generate_network_graph(G):
    # matplotlib 백엔드 설정
    matplotlib.use('Agg')
    
    # 한글 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 환경
    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rc('font', family=font_name)
    
    # 그래프 크기 조절
    plt.figure(figsize=(8, 6))
    
    # 노드 크기 설정
    node_size = [G.degree(node) * 200 for node in G.nodes()]
    
    # 엣지 굵기 설정
    edge_width = [G[u][v]['weight'] * 0.3 for u, v in G.edges()]
    
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
    
    # base64로 인코딩
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    plt.close()
    
    return graphic 