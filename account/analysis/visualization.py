import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
from io import BytesIO
import base64
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples

# 로깅 설정
logger = logging.getLogger(__name__)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def evaluate_model_performance(embeddings, labels, sil_score):
    """
    클러스터링 모델 성능 평가 및 시각화
    """
    try:
        plt.figure(figsize=(12, 6))
        
        # 1. 실루엣 분석 그래프
        plt.subplot(1, 2, 1)
        
        # y축 범위 설정
        y_lower = 10
        n_clusters = len(set(labels))
        
        # 클러스터별 실루엣 값 계산
        silhouette_vals = silhouette_samples(embeddings, labels, metric='cosine')
        
        # 클러스터별로 실루엣 값 시각화
        for i in range(n_clusters):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            # 클러스터별 색상 설정 (초록색/회색 계열)
            color = '#4CAF50' if i == 0 else '#757575'
            alpha = 0.7 if i == 0 else 0.5
            
            # 실루엣 값 채우기
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color,
                            alpha=alpha)
            
            # 클러스터 레이블 표시
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, 
                    f'클러스터 {i+1}\n(크기: {size_cluster_i})')
            y_lower = y_upper + 10
        
        # 평균 실루엣 점수 수직선
        plt.axvline(x=sil_score, color='red', linestyle='--',
                   label=f'평균 실루엣 점수: {sil_score:.3f}')
        
        plt.title('클러스터별 실루엣 분석', pad=20)
        plt.xlabel('실루엣 계수')
        plt.ylabel('클러스터 레이블')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # 2. 클러스터 크기 분포 (파이 차트)
        plt.subplot(1, 2, 2)
        
        # 클러스터별 크기 계산
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        colors = ['#4CAF50', '#757575']  # 초록색/회색 계열
        
        plt.pie(cluster_sizes,
               labels=[f'클러스터 {i+1}\n({size}개)' for i, size in enumerate(cluster_sizes)],
               colors=colors,
               autopct='%1.1f%%',
               startangle=90)
        
        plt.title('클러스터 크기 분포', pad=20)
        
        # 전체 레이아웃 조정
        plt.suptitle('연관어 클러스터링 품질 평가', 
                    fontsize=16, y=1.05)
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
        logger.error(f"클러스터링 성능 평가 시각화 중 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_network_graph(G, node_clusters=None):
    """네트워크 그래프 시각화 (클러스터 정보 포함)"""
    try:
        plt.figure(figsize=(16, 12), facecolor='none')
        ax = plt.gca()
        ax.set_facecolor('none')
        
        # 노드 위치 계산 (매번 다른 레이아웃)
        pos = nx.spring_layout(G, k=2, iterations=100, 
                             seed=np.random.randint(1, 1000))
        
        # 노드 크기와 색상 계산
        node_sizes = []
        node_colors = []
        nodes = list(G.nodes())
        
        for node in nodes:
            # 노드 크기: 연결 수 + 가중치
            size = (G.degree(node) + 1) * 500
            node_sizes.append(size)
            
            # 노드 색상: 클러스터별 구분
            if node_clusters is not None:
                cluster = node_clusters[nodes.index(node)]
                color = '#4CAF50' if cluster == 0 else '#757575'
            else:
                color = '#1f77b4'
            node_colors.append(color)
        
        # 엣지 그리기
        edge_weights = nx.get_edge_attributes(G, 'weight')
        if edge_weights:
            min_weight = min(edge_weights.values())
            max_weight = max(edge_weights.values())
            
            sorted_edges = sorted(G.edges(data=True), 
                                key=lambda x: x[2]['weight'])
            
            for u, v, data in sorted_edges:
                weight = data['weight']
                if min_weight == max_weight:
                    alpha = 0.6
                    width = 4
                else:
                    alpha = 0.3 + 0.6 * (weight - min_weight) / (max_weight - min_weight)
                    width = 1 + 8 * (weight - min_weight) / (max_weight - min_weight)
                
                nx.draw_networkx_edges(G, pos,
                                     edgelist=[(u, v)],
                                     width=width,
                                     edge_color='purple',
                                     alpha=alpha,
                                     style='solid')
        
        # 노드와 레이블 그리기
        nx.draw_networkx_nodes(G, pos, 
                             node_size=node_sizes,
                             node_color=node_colors,
                             alpha=0.7)
        
        nx.draw_networkx_labels(G, pos,
                              font_family=plt.rcParams['font.family'],
                              font_size=12,
                              font_weight='bold',
                              font_color='white')
        
        plt.title('연관어 키워드 네트워크',
                 fontdict={'family': plt.rcParams['font.family'],
                          'size': 24,
                          'weight': 'bold'},
                 pad=50)
        
        plt.axis('off')
        plt.tight_layout(pad=20)
        
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
        logger.error(f"네트워크 그래프 생성 중 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def visualize_performance_metrics(metrics_list):
    """성능 메트릭 시각화"""
    try:
        if not metrics_list:
            logger.warning("성능 메트릭 데이터가 없습니다.")
            return None
            
        # 리스트를 DataFrame으로 변환
        df = pd.DataFrame(metrics_list)
        
        # 시간순 정렬
        df = df.sort_values('timestamp')
        
        plt.figure(figsize=(15, 10), facecolor='white')
        
        # 처리 시간 추이
        plt.subplot(2, 2, 1)
        plt.plot(df['timestamp'], df['processing_time'], 'b-', 
                linewidth=2, color='#2196F3')
        plt.title('처리 시간 추이', fontsize=12, pad=10)
        plt.xlabel('분석 시간', fontsize=10)
        plt.ylabel('처리 시간(초)', fontsize=10)
        plt.xticks(rotation=45)
        
        # 메모리 사용량 추이
        plt.subplot(2, 2, 2)
        plt.plot(df['timestamp'], df['memory_usage'], '-', 
                linewidth=2, color='#4CAF50')
        plt.title('메모리 사용량 추이', fontsize=12, pad=10)
        plt.xlabel('분석 시간', fontsize=10)
        plt.ylabel('메모리(MB)', fontsize=10)
        plt.xticks(rotation=45)
        
        # 키워드 수 분포
        plt.subplot(2, 2, 3)
        plt.hist(df['keyword_count'], bins=20, color='#42A5F5', 
                edgecolor='white')
        plt.title('키워드 수 분포', fontsize=12, pad=10)
        plt.xlabel('키워드 수', fontsize=10)
        plt.ylabel('빈도', fontsize=10)
        
        # 실루엣 점수 추이
        plt.subplot(2, 2, 4)
        plt.plot(df['timestamp'], df['silhouette_score'], '-', 
                linewidth=2, color='#F44336')
        plt.title('실루엣 점수 추이', fontsize=12, pad=10)
        plt.xlabel('분석 시간', fontsize=10)
        plt.ylabel('실루엣 점수', fontsize=10)
        plt.xticks(rotation=45)
        
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
        return None
  