import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
from io import BytesIO
import base64
import logging
import pandas as pd

# 로깅 설정
logger = logging.getLogger(__name__)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def generate_network_graph(G):
    """네트워크 그래프 시각화"""
    try:
        # 투명한 배경으로 figure 생성
        plt.figure(figsize=(16, 12), facecolor='none')
        
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
        plt.tight_layout(pad=20)
        
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

def visualize_performance_metrics(performance_metrics=None):
    """성능 메트릭 시각화"""
    try:
        # 데이터가 없을 때 기본 데이터 생성
        if not performance_metrics:
            # 기본 성능 데이터 생성
            default_metrics = {
                'processing_time': [0.5, 0.6, 0.4, 0.5],
                'memory_usage': [100, 110, 95, 105],
                'keyword_count': [15, 18, 12, 16],
                'silhouette_score': [0.7, 0.75, 0.8, 0.85]
            }
            df = pd.DataFrame(default_metrics)
        else:
            df = pd.DataFrame(performance_metrics)

        # 한글 폰트 설정
        try:
            font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
            font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        except:
            logger.warning("기본 폰트를 사용합니다.")
            plt.rcParams['font.family'] = 'Malgun Gothic'

        # seaborn 스타일 대신 matplotlib 내장 스타일 사용
        plt.style.use('bmh')
        
        plt.figure(figsize=(15, 10), facecolor='white')
        
        # 전체 그래프 스타일 설정
        plt.rcParams['axes.facecolor'] = '#f8f9fa'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.color'] = '#cccccc'
        
        # 처리 시간 추이
        plt.subplot(2, 2, 1)
        plt.plot(range(len(df)), df['processing_time'], 'b-', linewidth=2, color='#2196F3')
        plt.title('처리 시간 추이', fontsize=12, pad=10)
        plt.xlabel('분석 횟수', fontsize=10)
        plt.ylabel('처리 시간(초)', fontsize=10)
        
        # 메모리 사용량 추이
        plt.subplot(2, 2, 2)
        plt.plot(range(len(df)), df['memory_usage'], '-', linewidth=2, color='#4CAF50')
        plt.title('메모리 사용량 추이', fontsize=12, pad=10)
        plt.xlabel('분석 횟수', fontsize=10)
        plt.ylabel('메모리(MB)', fontsize=10)
        
        # 키워드 수 분포
        plt.subplot(2, 2, 3)
        plt.hist(df['keyword_count'], bins=20, color='#42A5F5', edgecolor='white')
        plt.title('키워드 수 분포', fontsize=12, pad=10)
        plt.xlabel('키워드 수', fontsize=10)
        plt.ylabel('빈도', fontsize=10)
        
        # 실루엣 점수 추이
        plt.subplot(2, 2, 4)
        valid_scores = df[df['silhouette_score'] > 0]['silhouette_score']
        if not valid_scores.empty:
            plt.plot(range(len(valid_scores)), valid_scores, '-', 
                    linewidth=2, color='#F44336')
            plt.title('실루엣 점수 추이', fontsize=12, pad=10)
            plt.xlabel('분석 횟수', fontsize=10)
            plt.ylabel('실루엣 점수', fontsize=10)
        else:
            plt.text(0.5, 0.5, '유효한 실루엣 점수 없음', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes)
        
        plt.tight_layout(pad=3.0)
        
        # 이미지 저장
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close('all')
        
        return base64.b64encode(image_png).decode('utf-8')
        
    except Exception as e:
        logger.error(f"성능 메트릭 시각화 중 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None 