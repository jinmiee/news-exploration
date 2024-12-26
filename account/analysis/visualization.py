import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
import networkx as nx
from io import BytesIO
import base64
import logging
import pandas as pd
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def generate_network_graph(G, node_clusters=None):
    """네트워크 그래프 시각화"""
    try:
        plt.figure(figsize=(16, 12), facecolor='none')
        ax = plt.gca()
        ax.set_facecolor('none')

        
        # 노드 위치 계산 (매번 다른 레이아웃)
        pos = nx.spring_layout(G, k=2, iterations=50,  # iterations 줄임
                             seed=42)  # 고정된 seed 사용
        
        # 노드 크기와 색상 계산
        node_sizes = []
        nodes = list(G.nodes())
        
        for node in nodes:
            # 노드 크기: 연결 수 + 가중치
            size = (G.degree(node) + 1) * 700
            node_sizes.append(size)
        
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
                             node_color='#1f77b4',
                             alpha=0.7)
        
        nx.draw_networkx_labels(G, pos,
                              font_family=plt.rcParams['font.family'],
                              font_size=12,
                              font_weight='bold',
                              font_color='white')
        
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
  