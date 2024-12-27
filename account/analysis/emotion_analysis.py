import re
import platform

import matplotlib.font_manager
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from konlpy.tag import Okt
from collections import Counter
import matplotlib.font_manager as fm
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import pandas as pd
import platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
import networkx as nx
from io import BytesIO
import base64
import logging

# 한글 폰트 경로 설정 (NanumGothic 사용)
if platform.system() == 'Windows':
    font_path = "C:/Windows/Fonts/malgun.ttf"
else:  # Linux
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_prop = matplotlib.font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = font_prop.get_name()

custom_model_path = "account/custom_model"

# 전역 변수로 설정하여 재사용
okt = Okt()
sentiment_analysis_pipeline = pipeline("text-classification", model=custom_model_path, truncation=True, padding=True, max_length=512)


# 전역 변수로 설정하여 재사용

# 불용어 리스트 (필요시 확장 가능)
stopwords = [
    '이', '그', '저', '것', '들', '은', '는', '이', '가', '에', '과', '와', '도', '으로', '에게', '으로', '다', '이다', '있다', '없다', '뿌리',
    '모자', '대공', '야간',
                '대통령', '국민', '민주당', '국회', '방송', '국가', '군장', '미필', '통해', '조사', '가정', '나라', '국힘', '뉴스', '윤석열', '독재정',
    '밤샘', '트랙터', '때문', '강원도', '국회의원',
    '당시', '때문', '정치인', '세력', '이재명', '개똥', '일본', '시간', '윤거', '지금', '북파공작원', '이나라', '한국', '경찰', '우리나라', '대한민국', '을지',
    '잘못', '조끼', '폭약',
    '준비', '이번', '오세훈', '인간성', '가슴', '가지', '머리', '당선', '누가', '전두환', '이유', '제대로', '정상', '하나', '기회', '다른', '그냥', '생명',
    '얼마나', '방첩', '오토바이',
    '동조자', '사령관', '가차', '아들', '대표', '판사', '영상', '로부터', '생각', '군인', '본보기', '다른', '한자', '도와방송사', '대충', '거나', '노래', '선택',
    '잠도', '자기', '신뢰',
    '하루하루', '방탄', '부역', '부터', '정청래', '팩트', '모조리', '우두머리', '운영', '정말', '큰일', '주의자', '눈물', '도외', '국정원', '인강성', '실화',
    '한미', '감옥',
    '집단', '일이', '수사권', '국가보안법', '사람', '위해', '사법', '일이', '수사', '피땀', '시한', '불면', '세로', '이제', '소름', '두창', '기득권', '희생',
    '두통', '대사관', '강원도',
    '필요', '인생', '젊은이', '경제', '건희', '무당파', '집중', '이제', '방송사', '엠비씨', '원인', '중공', '미국', '언론', '통일', '전면', '심판', '제공',
    '지배', '원인', '극우', '수송'
                      '참내', '언론', '잔당', '집행', '지령', '개정', '일신', '마음대로', '라차', '뒥일', '테러범', '진짜', '역시', '거기', '차려', '제목',
    '된거', '획책', '자식', '추운', '안해', '테이저건'
                                  '일련정종', '천수', '목숨', '집회', '투쟁', '지령', '일부', '차빼', '공산주의', '커녕', '김어준', '가쇼', '지역구',
    '라가', '강민국', '역자', '윤거니', '일상', '굥부부', '멀리',
    '방송국', '조항', '문재인', '정신', '날씨', '진주', '시민', '덩어리', '마지막', '전체', '위태롭네', '통화', '인간성', '입맞춤', '헌법재판소', '진행중', '장기판',
    '작전', '안나',
    '대사관', '추경호', '냄새', '관련', '필요', '근본', '축소', '진동', '당사', '이철우', '홍준표', '행정안전부', '권력', '유정복', '서울시', '비공개', '구역',
    '증거', '절대', '대구시',
    '경북', '절대', '촉각', '진행', '투표', '김병주', '헌법재판', '중단', '의원님', '조작', '너트', '본회의', '의문', '김진태', '깜빵', '그날', '전국', '특전사',
    '지연', '확정', '안위',
    '발표', '포함', '대책', '누구', '비상', '적극', '병사', '수방사', '실탄', '현행범', '완전', '꼼수', '장갑차', '호크', '주방사', '공수부대', '인천', '저격',
    '대책', '현재', '용총', '거부'
                      '산탄총', '선전', '검찰', '합동', '차량', '임시', '대북분노', '각성', '드론', '무슨', '인간', '기관총', '세계', '권총', '일련정종',
    '테이저건', '정보사', '재밍건', '무기', '상관', '수송',
    '시초', '겨누', '부모', '아첨', '바이든', '세상', '극형발악', '법적', '긴급', '아이', '우길', '보고', '분열', '산탄총', '참내', '박수', '교육', '시경',
    '타격', '장난', '야간', '법적', '유치원',
    '양심', '치인', '교육', '동참', '응석받이', '물정', '간주', '야간', '주제', '찬성', '치인', '버스', '옹호', '동조', '간주', '대북', '정당', '반성', '선포',
    '선제', '공부', '특수부대', '다음',
    '대형', '모두', '두번째', '자네', '바크', '정치가', '큰소리', '댓글', '삭제', '만드구', '엠빙신', '체포하라', '어디', '마련', '주드', '쌈질', '동원', '마당',
    '살상', '웃기', '읏기', '끼리', '선제',
    '과연', '소니', '머슴', '라면', "목소리", "기회", "기본", "과제", "주요", "대상", "관점", "판단", "처리", "변화", "상태", "원인", "결론", "기록", "목표",
    "목적", "분석", "방법",
    "방안", "실제", "결과", "점검", "연대", "조치", "회차", "전반", "정도", "위원", "방향", "측면", "요소", "통계", "수치", "규모", "성공", "지침", "기준",
    "대응", "보고", "연결",
    "발전", "형태", "속도", "정체", "조건", "국면", "결과", "수단", "연관", "자료",  "경우", "책임", "주제", "법적", "문서", "배경", "상황", "방안",
    "시각", "현황", "성명",
    "주장", "위기", "조정", "분배", "연결", "통합", "동기", "성격", "성립", "부문", "정렬", "법률", "정부", "기관", "조직", "제도", "입장", "연결", "구성",
    "핵심", "상정", "단체",
    "경로", "협정", "배경", "접근", "가능", "구분", "집행", "분류", "행동", "주체", "지원", "강화", "비전", "조건", "정부", "사회", "정치", "시스템", "정책",
    "법", "규제", "법안",
    "사람", "나라", "국민", "경제", "상황", "문제", "이유", "소식", "기사", "정보", "미디어", "뉴스", "주제", "항목", "기술", "이슈", "소리", "논의", "담론",
    "지식", "사건", "대응",
    "조치", "위원회", "회의", "전망", "측면", "단계", "대상", "방향", "기준", "장소", "사유", "시간", "부분", "사실", "기반", "연구", "분석", "조사", "결정",
    "결과", "결의", "집단",
    "대중", "목표", "계획", "문헌", "보고서", "내용", "수단", "환경", "발표", "구조", "대통령", "총리", "국회", "의회", "장관", "위기", "선거", "정당", "집행",
    "협상", "동의",
    "대표", "합의", "지시", "일정", "검토", "검사", "보고서", "의미", "목표", "기준", "자료", "선택", "집단", "분야", "항목", "의제", "관계", "결과", "사례",
    "조정", "법원",
    "선언", "지원", "시행", "논의", "참여", "기회", "협력", "대응", "단계", "회복", "의도", "진전", "자원", "기회", "프로세스", "법률", "경험", "결정", "절차",
    "조사", "문제",
    "리더십", "선택", "주장", "보고", "요구", "형식", "상상", "강화", "집합", "단체", "행동", "조합", "결합", "기능", "편안", "조건", "평가", "실현", "유형",
    "기반", "안정",
    "해석", "결합", "양상", "경로", "분석", "전략", "집합", "설명", "분배", "특성", "문제", "결합", "결정", "시도", "정리", "상황", "조정", "세부", "방향",
    "관리", "세부",
    "내용", "원칙", "문제", "핵심", "기능", "기술", "진행", "시점", "가능", "계획", "목표", "운영", "측정", "계획", "자세", "기획", "심사", "조정", "업무",
    "성립", "지침", "동향",
    "구성", "경과", "기술", "상황", "장면", "변화", "변수", "주요", "대상", "분석", "정세", "안정", "보도", "배경", "사건", "계획", "효과", "조치",
    "장소", "국면", "문제",
    "위기", "정리", "적용", "인식", "방안", "논란", "부분", "연구", "분석", "검토", "단계", "수준", "이해", "기록", "리스트", "절차", "구체화", "문서", "상징",
    "양상", "요소", "결과",
    "대책", "조정", "모델", "기대", "계획", "역할", "방법", "시기", "상관", "측정", "상세", "현실", "실제", "규모", "기초", "의사", "집행", "성격", "목표",
    "이상", "방향", "구체",
    "기준", "형식", "기록", "분류", "대상", "정리", "준비", "기능", "기회", "전망", "문헌", "책임", "기반", "결정", "주장", "처리", "보상", "성취", "진전",
    "형태", "검토", "시스템",
    "자원", "시도", "확대", "진행", "수단", "기획", "의도", "평가", "지원", "기능", "규정", "검사", "성과", "상장", "분위기", "조건", "규정", "설계", "내용",
    "실천", "프로세스",
    "단체", "협정", '경찰차', '소나', '차고', '제봅', '하나님', '수갑', '친미', '언제', '제작', '보더', '거부', '기고', '섬뜩', '조직체', '지네', '국기', '하라',
    '기고', '지지', '아침', '앵커', '범죄인',
    '비시', '반중', '지이미', '반드시', '보호', '포구', '윤건', '도피', '편파', '반중', '남태령', '한덕수', '지고', '궁행', '국짐', '재판관', '너희', '안녕',
    '어른', '헌재', '임명', '우리', '권한', '대행', '할배',
    '꼭두각시', '개인', '악덕', '등사', '권성동', '김건희', '권리', '공범', '리도', '총선', '곳간', '파면', '얘기', '통치행위', '반격', '재미', '윤통', '아든',
    '고라', '미래', '국민연금', '강제', '총선', '곳간', '중이',
    '파약', '국민연금', '당장', '어차피', '마감', '패스', '고라', '전부', '눈빛', '상대', '국짐', '아주', '토리', '선재', '즉시', '누리', '저런', '중앙', '기출',
    '업주', '역사', '정립', '환율', '투자', '노릇', '년후',
    '공연', '헌법재판관', '공연', '파악', '돼지', '취소', '쯧쯧', '법칙', '여우', '상의', '바로', '역사', '위헌', '평생', '한나라당', '어스', '수가', '민주주의',
    '빈부격차', '야당', '단독', '질질', '이야기', '바지',
    '괴물', '부정선거계엄', '미숙아', '서민', '나가야', '강제집행', '물풍선', '거머리', '감빵', '심화', '이전', '가수', '바로', '덕수', '침범', '쉬이', '마트',
    '변호', '자리', '국짐애', '독안', '중지', '불량품', '해결', '역풍',
    '찰떡', '다수', '건설', '개성', '발악', '마안', '이자', '진저리', '신세', '스리', '돌파', '뻑하', '포위', '수괴', '태풍', '비롯', '시대', '삼권', '갓집',
    '멸망', '형국', '오늘', '안중', '추찹', '규명', '시작', '여당', '어굴하',
    '나무늘보', '눈치', '한동훈', '이승환', '도움', '어이', '이상민', '이종인', '족속', '보지', '목격', '벨라', '절대로', '한민당', '양반', '주안', '행태', '먼저',
    '따위', '기도', '징글징글', '진상', '하자', '기계', '지지율', '이어도',
    '산이', '유인춘', '족속', '태산', '도움', '걱정', '신기록', '한민당', '절대로', '구미', '과감', '산이', '조해', '양반', '변호인', '도움', '잡범', '윤석렬',
    '김여사', '자유당', '공화당', '새누리당', '한탄', '동현', '분립',
    '모시', '입법', '고양이', '즉각', '민자당', '결집', '손해배상', '헌법', '축구', '조수진', '신한국당', '화합', '전주', '유인촌', '세금', '다시', '멸시', '두기',
    '침대', '원리', '그게', '행위', '원죄', '고도', '행사', '마비', '중도', '방조', '조범', '결집', '하루', '사쿠라', '용서', '견제', '분립', '분리', '도데',
    '왕권',"문재인", "윤석열", "이재명", "박근혜", "이명박", "김대중", "노무현", "김정은", "황교안", "안철수", "정세균", "홍준표", "김종인", "오세훈", "유시민", "유승민",
    "이낙연", "심상정", "권성동", "정청래", "조국", "이해찬", "김상희", "조정식", "김영춘", "김태년", "박지원", "강경화", "강기정", "박주민", "조승진", "최재형", "이준석", "이홍기",
    "배현진", "김미애", "박상기", "한덕수", "이기우", "김영호", "홍익표", "양정철", "안민석", "조윤선", "권영세", "박민식", "홍정욱", "이호철", "김재원", "김승희", "이재오", "김두관",
    "김성곤", "이상민", "한기호", "문성혁", "최강욱", "윤호중", "김용민", "김성환", "한명숙", "임종석", "진영", "박영선", "서병수", "이동학", "전병헌", "이종걸", "정의당", "이동섭", "서영교",
    "송영길", "박성효", "이상화", "최성", "최용기", "안경률", "이상갑", "김형오", "김광진", "정동영", "김철수", "한상진", "이규섭", "백종원", "김찬래", "배진교", "강준현", "권혁열", "박용진",
    "김정연", "서정숙", "김영기", "박문호", "최성준", "서영선", "차영우", "장제원", "이철희", "김덕기", "조광한", "김경협", "강석호", "박수영", "이종화", "강훈식", "박성준", "양정은", "최상화",
    "김미경", "김현미", "김태호", "이재율", "김태영", "안호영", "황진호", "정진석", "박정", "정재희", "김건", "박영일", "조승환", "임상철", "이정헌", "윤미향", "김두리", "권오곤", "배덕환",
    "전상희", "정희용", "정재연", "정대용", "이혜경", "최성규", "박성일", "이원욱", "정양기", "김희경", "홍종학", "조훈현", "이광재", "양승태", "최진락", "황정아", "유기홍", "정병국", "정원섭",
    "최승희", "남경필", "박상수", "차성진", "김상기", "최재혁", "권경우", "김인순", "고진택", "오충환", "정승길", "김문수", "박형철", "남경수", "조은희", "김대기", "황명선", "김덕수", "박성기",
    "이혜인", "박성욱", "송건호", "정용기", "김정섭", "황경택", "한정애", "정만수", "이상헌", "황보승희", "정두언", "김명희", "정연주", "강동원", "이순형", "한정화", "강명희", "김철수", "박수미",
    "이주호", "김태광", "정인수", "송시호", "오규석", "전재희", "장재영", "김대영", "김재섭", "이정희", "정연우", "김영진", "이진희", "이종훈", "정용선", "양창일", "이순호", "임태훈", "김경희",
    "장문철", "박은수", "박정래", "윤선애", "김효석", "정원상", "정병원", "정창균", "김봉교", "권형호", "백병기", "김상일", "이기성", "이시대", "송수영", "최경만", "김재민", "배병석", "이덕상",
    "윤리영", "정윤정", "유성엽", "최병욱", "권영선", "최진영", "박효민", "강남", "유상호", "최병훈", "김혜옥", "정희재","황의조", "이강인", "레베카 쳅테게이", "기성용", "김하성",
    '조배숙', '수작', '해산', '북도', '김종인', '민자당', '원죄', '민주당', '용산', '의원', '아스팔트', '정답', '민주',
    '별로', '어찌', '무대', '의원', '가요', '지름길', '무속', '니놈', '천년', '만행', '행정', '주가조작', '생선', '사람과', '산다', '전복', '중국', '해먹',
    '거리', '근원', '개념', '남발', '관심', '별로', '어찌', '공산', '갈래',
    '국무위원', '기간', '달력', '익산', '사람과', '볼때',  '늬적거림', '신민주공화당', '떼쟁이', '마구잡이', '주사파', '굿짐', '발생', '내년', '갈래', '공산',
    '인방', '행사', '마비', '국민주권', '지름길', '여생', '수백', '자가',
    '표현', '면상', '생중계', '일치', '청구인', '담보', '기각', '가계', '민정당', '성동', '텅텅', '성동', '계속', '맙시', '자가', '아치', '지체', '미래세',
    '서로', '과학', '사나', '제일', '대해', '맙시', '장영', '표현의자유', '궁금하',
    '방어', '이바', '아난다', '중진', '텅텅', '착각', '마구', '지랄', '아예', '가중', '늑대', '단어', '관상', '먹듯', '중도', '단어', '각종', '것입', '나경원',
    '아기', '궁금하', '입법부', '남아', '한국은행', '두고두고', '사해', '대항',
    '다라', '보기', '악날', '법체계', '외화', '투자자', '외국', '사용', '님이시', '꾼', '윤상현', '외화', '모든', '궁금하', '자백', '보장', '기금', '가관',
    '보루', '모든', '수도', '응시', '말라가', '응시', '수도', '계략', '궁금하', '어쩌구',
    '강릉', '자격', '마음', '한쪽', '알바생', '칭구', '탁핵', '말장난', '명령', '스폰', '가야', '지도', '선관위', '노무현', '소령', '흉터',
    '이간', '부부', '우선', '동분서주', '취재', '황하', '권중현', '끼리끼리', '성폭행',
    '민낯', '처음', '세기', '말로', '짓거리', '우선', '국사', '방빼', '딩기', '먹물', '자유민주', '졸개', '공산당', '노태악', '고캐비넛', '선철', '윤수',
    '엉망', '대안',  '궤멸하', '시편', '취재', '상식', '양쪽', '이', '그', '저', '것', '들', '은', '는', '이', '가', '에', '과', '와', '도',
    '으로', '에게','만들었구만','내렸다','퍼마시','그려고도','않는말','받게','드리고','믿고','보다','똑바로','만들어','뽑은','도대체','되는거','말아먹고','되는',
    '으로', '다', '이다', '있다', '없다', '뿌리', '모자', '대공', '야간', "가", "가까스로", "가령", "각", "각각", "각자", "각종", "갖고말하자면",
    "같다", "같이", "개의치않고", "거니와", "거바", "거의", "것", "것과 같이", "것들",
    "게다가", "게우다", "겨우", "견지에서", "결과에 이르다", "결국", "결론을 낼 수 있다", "겸사겸사",
    "고려하면", "고로", "곧", "공동으로", "과", "과연", "관계가 있다", "관계없이", "관련이 있다",
    "관하여", "관한", "관해서는", "구", "구체적으로", "구토하다", "그", "그들", "그때", "그래",
    "그래도", "그래서", "그러나", "그러니", "그러니까", "그러면", "그러므로", "그러한즉", "그런 까닭에",
    "그런데", "그런즉", "그럼", "그럼에도 불구하고", "그렇게 함으로써", "그렇지", "그렇지 않다면",
    "그렇지 않으면", "그렇지만", "그렇지않으면", "그리고", "그리하여", "그만이다", "그에 따르는",
    "그위에", "그저", "그중에서", "그치지 않다", "근거로", "근거하여", "기대여", "기점으로", "기준으로",
    "기타", "까닭으로", "까악", "까지", "까지 미치다", "까지도", "꽈당", "끙끙", "끼익", "나", "나머지는",
    "남들", "남짓", "너", "너희", "너희들", "네", "넷", "년", "논하지 않다", "놀라다", "누가 알겠는가",'건가',
    "누구", "다른", "다른 방면으로", "다만", "다섯", "다소", "다수", "다시 말하자면", "다시말하면", "다음",
    "다음에", "다음으로", "단지", "답다", "당신", "당장", "대로 하다", "대하면", "대하여", "대해 말하자면",
    "대해서", "댕그", "더구나", "더군다나", "더라도", "더불어", "더욱더", "더욱이는", "도달하다", "도착하다",
    "동시에", "동안", "된바에야", "된이상", "두번째로", "둘", "둥둥", "뒤따라", "뒤이어", "든간에", "들",
    "등", "등등", "딩동", "따라", "따라서", "따위", "따지지 않다", "딱", "때", "때가 되어", "때문에", "또",
    "또한", "뚝뚝", "라 해도", "령", "로", "로 인하여", "로부터", "로써", "륙", "를", "마음대로", "마저", "마저도",
    "마치", "막론하고", "만 못하다", "만약", "만약에", "만은 아니다", "만이 아니다", "만일", "만큼", "말하자면",'받았나','북중',
    "말할것도 없고", "매", "매번", "메쓰겁다", "몇", "모", "모두", "무렵", "무릎쓰고", "무슨", "무엇", "무엇때문에",
    "물론", "및", "바꾸어말하면", "바꾸어말하자면", "바꾸어서 말하면", "바꾸어서 한다면", "바꿔 말하면", "바로", "바와같이",
    "밖에 안된다", "반대로", "반대로 말하자면", "반드시", "버금", "보는데서", "보다더", "보드득", "본대로", "봐", "봐라",'제판','않은','전광훈','갔다',
    "부류의 사람들", "부터", "불구하고", "불문하고", "붕붕", "비걱거리다", "비교적", "비길수 없다", "비로소", "비록",
    "비슷하다", "비추어 보아", "비하면", "뿐만 아니라", "뿐만아니라", "뿐이다", "삐걱", "삐걱거리다", "사", "삼", "상대적으로 말하자면",
    "생각한대로", "설령", "설마", "설사", "셋", "소생", "소인", "솨", "쉿", "습니까", "습니다", "시각", "시간", "시작하여",
    "시초에", "시키다", "실로", "심지어", "아", "아니", "아니나다를가", "아니라면", "아니면", "아니었다면", "아래윗", "아무거나",
    "아무도", "아야", "아울러", "아이", "아이고", "아이구", "아이야", "아이쿠", "아하", "아홉", "안 그러면", "않기 위하여",
    "않기 위해서", "알 수 있다", "알았어", "앗", "앞에서", "앞의것", "야", "약간", "양자", "어", "어기여차", "어느",'가지마','믿고'
    "어느 년도", "어느것", "어느곳", "어느때", "어느쪽", "어느해", "어디", "어때", "어떠한", "어떤", "어떤것", "어떤것들",
    "어떻게", "어떻해", "어이", "어째서", "어쨋든", "어쩌라고", "어쩌면", "어쩌면 해도", "어쩌다", "어쩔수 없다", "어찌",
    "어찌됏든", "어찌됏어", "어찌하든지", "어찌하여", "언제", "언젠가", "얼마", "얼마 안 되는 것", "얼마간", "얼마나", "얼마든지",
    "얼마만큼", "얼마큼", "엉엉", "에", "에 가서", "에 달려 있다", "에 대해", "에 있다", "에 한하다", "에게", "에서", "여", "여기",
    "여덟", "여러분", "여보시오", "여부", "여섯", "여전히", "여차", "연관되다", "연이서", "영", "영차", "옆사람", "예", "예를 들면",
    "예를 들자면", "예컨대", "예하면", "오", "오로지", "오르다", "오자마자", "오직", "오호", "오히려", "와", "와 같은 사람들",
    "와르르", "와아", "왜", "왜냐하면", "외에도", "요만큼", "요만한 것", "요만한걸", "요컨대", "우르르", "우리", "우리들", "우선", "우에 종합한것과같이",
    "운운", "월", "위에서 서술한바와같이", "위하여", "위해서", "윙윙", "육", "으로", "으로 인하여", "으로서", "으로써", "을", "응", "응당",
    "의", "의거하여", "의지하여", "의해", "의해되다", "의해서", "이", "이 되다", "이 때문에", "이 밖에", "이 외에", "이 정도의", "이것",
    "이곳", "이때", "이라면", "이래", "이러이러하다", "이러한", "이런", "이럴정도로", "이렇게 많은 것", "이렇게되면", "이렇게말하면",
    "이렇구나", "이로 인하여", "이르기까지", "이리하여", "이만큼", "이번", "이봐", "이상", "이어서", "이었다", "이와 같다", "이와 같은",
    "이와 반대로", "이와같다면", "이외에도", "이용하여", "이유만으로", "이젠", "이지만", "이쪽", "이천구", "이천육", "이천칠", "이천팔",
    "인 듯하다", "인젠", "일", "일것이다", "일곱", "일단", "일때", "일반적으로", "일지라도", "임에 틀림없다", "입각하여", "입장에서", "잇따라", "있다",
    "자", "자기", "자기집", "자마자", "자신", "잠깐", "잠시", "저", "저것", "저것만큼", "저기", "저쪽", "저희", "전부",
    "전자", "전후", "점에서 보아", "정도에 이르다", "제", "제각기", "제외하고",
    "조금", "조차", "조차도", "졸졸", "좀", "좋아", "좍좍", "주룩주룩", "주저하지 않고",
    "줄은 몰랏다", "줄은모른다", "중에서", "중의하나", "즈음하여", "즉", "즉시", "지든지",'연합뉴스','싯점','뺏고','가라','하지말고','불러','되주는'
    "지만", "지말고", "진짜로", "쪽으로", "차라리", "참", "참나", "첫번째로", "쳇", "총적으로",'제발','바를','보네','않다',
    "총적으로 말하면", "총적으로 보면", "칠", "콸콸", "쾅쾅", "쿵", "타다", "타인", "탕탕",'하구도',
    "토하다", "통하여", "툭", "퉤", "틈타", "팍", "팔", "퍽", "펄렁", "하", "하게될것이다",'몰아서','셰셰','극상'
    "하게하다", "하겠는가", "하고 있다", "하고있었다", "하곤하였다", "하구나", "하기 때문에",'놀랍지도','내렸다'
    "하기 위하여", "하기는한데", "하기만 하면", "하기보다는", "하기에", "하나", "하느니",
    "하는 김에", "하는 편이 낫다", "하는것도", "하는것만 못하다", "하는것이 낫다", "하는바",'가가','이준석','보네','업자','매일',
    "하더라도", "하도다", "하도록시키다", "하도록하다", "하든지", "하려고하다", "하마터면",'먹고','하는거지','나도','하시겠다','받아','되주는'
    "하면 할수록", "하면된다", "하면서", "하물며", "하여금", "하여야", "하자마자", "하지 않는다면",'맞습니다','받아'
    "하지 않도록", "하지마", "하지마라", "하지만", "하하", "한 까닭에", "한 이유는", "한 후",'무능','없어져','건지', '들을','시켜','하네','자체','않나','막아야','하세요','마세요','허락'
    "한다면", "한다면 몰라도", "한데", "한마디", "한적이있다", "한켠으로는", "한항목", "할 따름이다", '김용현','해라','하는거냐','하지','하는데','하는거야','하지','먹고'
    "할 생각이다", "할 줄 안다", "할 지경이다", "할 힘이 있다", "할때", "할만하다", "할망정",'했는데','할려고','했나','헤헤','했는지','해야','겁니다','자는',
    "할뿐", "할수있다", "할수있어", "할줄알다", "할지라도", "할지언정", "함께", "해도된다", "해도좋다",'밝혔다','쥴리','해야'
    "해봐요", "해서는 안된다", "해야한다", "해요", "했어요", "향하다", "향하여", "향해서", "허",'하려고','숨어서','끌어내라','하는게',
    "허걱", "허허", "헉", "헉헉", "헐떡헐떡", "형식으로 쓰여", "혹시", "혹은", "혼자", "훨씬",'시켜라','잡아','법대','변호사',
    "휘익", "휴", "흐흐", "흥", "힘입어",'하는','하냐','하다','하면','지가','했다','버거','나오면','된다','한다','하고','해서','들이','나온다','끌어내','하는게','떠들어',
    '교회', '해도','되면','농민','자급','가보면','올리고','버렸네','버는게','합시다','한다고','식당','아파트','보면','오천만','래야','내리고','이창용','합니다','먹는','보이콧',
    '하기','해야지','하는거','하게','저걸','유승','기리','회견','뭔가','하기'
]

# 이모티콘과 특수문자 제거 전처리 함수
import re

def preprocess_text(text):
    # 줄바꿈을 공백으로 처리
    text = text.replace('\n', ' ').replace('\r', ' ')
    # 유니코드 이모티콘 제거
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # 특수문자 및 기호 한번에 제거
    text = re.sub(r'[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]+', '', text)
    # 영어 알파벳 제거 (대소문자 모두)
    text = re.sub(r'[a-zA-Z]+', '', text)
    # 반복 제거 (한글의 두 글자 이상의 반복 문자)
    text = re.sub(r'[ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄻㅆㅉㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅢㅞ]{2,}', '', text)
    # 모든 숫자 제거
    text = re.sub(r'\d+', '', text)
    # 양쪽 끝 불필요한 공백 제거
    text = text.strip()
    return text

@lru_cache(maxsize=1000)
def extract_nouns_and_verbs(text):
    # 명사 및 동사 추출
    pos_tags = okt.pos(text)  # 품사 태깅
    return [word for word, tag in pos_tags if (tag == 'Noun' or tag == 'Verb') and word not in stopwords and len(word) > 1]  # 불용어 및 1글자 단어 제거

# TF-IDF 분석을 통한 중요 단어 추출 함수
def analyze_tfidf(comments):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform(comments)

    # TF-IDF 값과 단어를 매핑
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # 단어와 TF-IDF 값을 매칭하여 정렬
    word_score_pairs = list(zip(feature_names, tfidf_scores))
    sorted_word_scores = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)

    return sorted_word_scores[:100]  # 상위 100개의 중요 단어 반환

negative_keywords = ['악의', '내란', '범죄', '위험', '폭력', '위협', '비난', '불안', '혐오','대한민국','쓰레기','건전','단합','안보','폭망','선동', '전쟁','가짜 뉴스','흥청망청'
,'반역','편파','처벌','구속','극우','거짓','독재정','좌파','우파','살인','위해','개똥','반대','시위','깜빵','빵꾸','외세','남탓','공범','분노','충돌','세력','미치광이','견찰'
'비리','쿠데타','독재','거짓말','소굴','비상'  ,'원흉','이면','매국노','교주','우두머리','무속','주술','해체','부정선거','친일' '청산','가짜','극형','강제','충격','두목','탄핵','싸움'
,'문제','사이코','악마','관성','반성','무기징역','비리','친일파','초래','좌빨','지연','마비','처리','역적','잡놈','쑥대밭','부정','패거리','정신병','사냥개','카르텔','공포','감방'
,'공격','공모','간첩','민낯', '교활','징글징글','악인','공산당','교통체증','대가리','규탄','핑계','경고','스트레스','우려','걱정','무시','괴물','짓거리','비상계엄','계엄','친북','술집','룸싸롱'
  ,"발전",'일베','프레임' , '사욕', '막장', '전복', '수준', '책임','개미' ,'광기', '악질','정지','긴급','무지','난동','본색','돼지','기만','한계','알콜중독자','거덜','강원랜드','마담','교활','기적',
          '처단','사이비','최후',   '과세', '포기','추잡','죽어나','투기','기레기','마구잡이', '악인', '난장판',"문제"]

neutral_keywords = ['그냥', '모르겠다', '일상', '보통', '그렇다', '어쩔 수 없다', '무난', '그런가요', '이 정도','종교','교회','십자가','벼농사','달러','경기','금리','유지','하락','인상','기자회견'
                    ,'집권','언론사','의장','일당','여야','전라도','사태','대선','북한','효력']

positive_keywords = ['행복', '축하', '사랑', '환희', '희망', '긍정', '감사', '성공', '좋다', '훌륭하다', '즐겁다', '축복', '용기', '찬사','지지','지지자','안보'
                     ,'동참','건전','나라','똘똘','허가','찬성','중용','진실','열사','민주주의','안정','수호','비폭력','자유','잡범' ,'천국']

# 감정 분석 함수 (긍정, 부정, 중립)
def analyze_sentiment(word):
    try:
        # 중립, 긍정, 부정 키워드 체크
        if any(keyword in word for keyword in neutral_keywords):
            return '중립'
        elif any(keyword in word for keyword in positive_keywords):
            return '긍정'
        elif any(keyword in word for keyword in negative_keywords):
            return '부정'

        # 감정 분석 모델로 예측
        result = sentiment_analysis_pipeline(word)[0]
        label = result['label']  # '1 star', '2 stars', ..., '5 stars'

        # 별점을 -2 ~ 2로 변환
        star_to_label = {
            '1 star': 0,
            '2 stars': 1,
            '3 stars': 2,
            '4 stars': 3,
            '5 stars': 4
        }
        sentiment_score = star_to_label.get(label, 0)  # 기본값 0(중립)

        # 점수를 부정(-1 이하), 중립(0), 긍정(1 이상)으로 단순화
        if sentiment_score <= 1:
            return '부정'  # 부정
        elif sentiment_score == 2:
            return '중립'  # 중립
        else:
            return '긍정'  # 긍정
    except Exception as e:
        print(f"감정 분석 오류: {e}")
        return '중립'  # 오류 발생 시 중립 반환

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def save_bubble_chart_with_tfidf(comments, save_path='media/analysis_results/'):
    """
    댓글을 받아 TF-IDF 분석을 통해 중요한 단어를 추출하고,
    감정 분석 후 버블 차트를 생성하여 Base64로 반환합니다.
    """
    # 1. 댓글 전처리 (불필요한 문자 제거 등)
    preprocessed_comments = [preprocess_text(comment) for comment in comments]

    # 2. 명사와 동사만 추출
    extracted_words = []
    for comment in preprocessed_comments:
        nouns_and_verbs = extract_nouns_and_verbs(comment)  # 명사와 동사 추출
        extracted_words.append(" ".join(nouns_and_verbs))  # 추출한 명사와 동사를 결합

    # 3. TF-IDF 분석을 통한 중요한 단어 추출
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform(extracted_words)

    # TF-IDF 값과 단어를 매핑
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # 단어와 TF-IDF 값을 매칭하여 정렬
    word_score_pairs = list(zip(feature_names, tfidf_scores))
    sorted_word_scores = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)

    # 상위 100개의 중요 단어 추출 (TF-IDF에 의해 중요도가 높은 단어들)
    top_words = sorted_word_scores[:100]
    top_words_dict = {word: score for word, score in top_words}

    # 4. 각 단어의 빈도 계산 (빈도수 기반으로 크기 설정)
    word_counts = Counter()
    for comment in extracted_words:
        for word in comment.split():
            if word in top_words_dict:  # TF-IDF에서 나온 단어만 카운트
                word_counts[word] += 1

    # 빈도수 기준 상위 10개의 단어 추출
    most_frequent_words = word_counts.most_common(10)
    selected_words = {word: count for word, count in most_frequent_words}

    # 5. 감정 분석 결과 수집 및 색상 매핑
    sentiment_colors = {
        '긍정': '#5745e9',  # 긍정은 파란색
        '중립': '#ffc75a',  # 중립은 회색
        '부정': '#ff6f6f'  # 부정은 빨간색
    }
    word_color_mapping = {}
    sentiment_results = {
        '긍정': [],
        '중립': [],
        '부정': []
    }

    for word in top_words_dict.keys():
        sentiment = analyze_sentiment(word)  # 해당 단어의 감정 분석
        word_color_mapping[word] = sentiment_colors[sentiment]  # 색상 지정
        sentiment_results[sentiment].append(word)  # 감정 결과 분류

    # 상위 10개 단어와 관련 색상 필터링
    top10_word_counts = {word: selected_words[word] for word in selected_words.keys()}
    top10_word_color_mapping = {word: word_color_mapping[word] for word in selected_words.keys()}

    # 6. 버블 차트 생성
    def create_bubble_chart(word_counts, selected_word_color_mapping):
        """
        단어 빈도와 감정 색상에 기반하여 버블 차트를 생성합니다.
        """
        words = list(word_counts.keys())
        frequencies = list(word_counts.values())
        colors = [selected_word_color_mapping[word] for word in words]

        # 반지름 계산
        radii = [np.sqrt(freq) / 10 for freq in frequencies]

        # 원 배치 알고리즘 구현
        def pack_circles_from_center(radii):
            positions = []  # 각 원의 중심 좌표 저장
            positions.append((0, 0))  # 첫 번째 원은 중심에 배치

            for i, r in enumerate(radii[1:], start=1):  # 두 번째 원부터 배치
                placed = False
                angle = 0  # 배치할 각도
                radius_step = 0.1  # 중심에서 점진적으로 확장
                current_radius = radii[0] + r  # 중심에서 최소 거리 설정

                while not placed:
                    angle += np.pi / 180  # 각도를 조금씩 회전하며 위치 찾기
                    x = np.cos(angle) * current_radius
                    y = np.sin(angle) * current_radius
                    candidate_position = (x, y)

                    # 다른 원들과 충돌하지 않는지 확인
                    collision = False
                    for j, (px, py) in enumerate(positions):
                        d = np.linalg.norm(np.array(candidate_position) - np.array((px, py)))
                        if d < radii[j] + r:  # 충돌 발생
                            collision = True
                            break

                    if not collision:
                        positions.append(candidate_position)
                        placed = True
                    elif angle >= 2 * np.pi:  # 한 바퀴 돌았다면 중심 반지름 증가
                        angle = 0
                        current_radius += radius_step

            return positions

        # 원의 위치 계산
        positions = pack_circles_from_center(radii)

        # 시각화
        plt.figure(figsize=(10, 10))
        for i, (x, y) in enumerate(positions):
            circle = plt.Circle((x, y), radii[i], color=colors[i], alpha=0.6)
            plt.gca().add_patch(circle)
            plt.text(x, y, words[i], fontsize=10, ha='center', va='center', color='white')

        plt.axis('equal')
        plt.axis('off')
        plt.title("감성어 TOP 10", fontsize=16)

        # 차트를 Base64로 반환
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return base64_image

    # 버블 차트 생성
    bubble_chart_base64 = create_bubble_chart(top10_word_counts, top10_word_color_mapping)

    return bubble_chart_base64


def save_visualizations_with_tfidf(comments, save_path='media/analysis_results/'):
    """
    댓글을 받아 TF-IDF 분석을 통해 중요한 단어를 추출하고,
    감정 분석 후 워드클라우드와 파이차트를 생성하여 Base64로 반환합니다.
    """
    # 1. 댓글 전처리 (불필요한 문자 제거 등)
    preprocessed_comments = [preprocess_text(comment) for comment in comments]

    # 2. 명사와 동사만 추출
    extracted_words = []
    for comment in preprocessed_comments:
        nouns_and_verbs = extract_nouns_and_verbs(comment)  # 명사와 동사 추출
        extracted_words.append(" ".join(nouns_and_verbs))  # 추출한 명사와 동사를 결합

    # 3. TF-IDF 분석을 통한 중요한 단어 추출
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform(extracted_words)

    # TF-IDF 값과 단어를 매핑
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # 단어와 TF-IDF 값을 매칭하여 정렬
    word_score_pairs = list(zip(feature_names, tfidf_scores))
    sorted_word_scores = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)

    # 상위 100개의 중요 단어 추출 (TF-IDF에 의해 중요도가 높은 단어들)
    top_words = sorted_word_scores[:100]
    top_words_dict = {word: score for word, score in top_words}

    # 4. 각 단어의 빈도 계산 (빈도수 기반 워드클라우드 생성)
    word_counts = Counter()
    for comment in extracted_words:
        for word in comment.split():
            if word in top_words_dict:  # TF-IDF에서 나온 단어만 카운트
                word_counts[word] += 1

    # 5. 감정 분석 결과 수집
    sentiment_results = {
        '긍정': [],
        '중립': [],
        '부정': []
    }

    # 6. 각 댓글에 대해 단어별 감정 분석
    for comment in extracted_words:
        for word in comment.split():
            sentiment = analyze_sentiment(word)  # 각 단어에 대해 감정 분석 수행
            sentiment_results[sentiment].append(word)  # 감정 결과에 따라 분류

    # 7. 감정별 색상 맵 설정
    sentiment_colors = {
        '긍정': '#5745e9',  # 긍정은 파란색
        '중립': '#ffc75a',     # 중립은 회색
        '부정': '#ff6f6f'      # 부정은 빨간색
    }


    # 8. 단어와 해당 감정의 색상을 매핑
    word_color_mapping = {}
    for word in top_words_dict.keys():
        sentiment = analyze_sentiment(word)  # 해당 단어의 감정 분석
        word_color_mapping[word] = sentiment_colors[sentiment]  # 색상 지정

    def save_wordcloud_base64(word_count, word_color_mapping):
# 감정 분석 후 각 감정별 단어를 수집하여 워드클라우드와 파이차트를 생성하는 함수

        """
        주어진 단어 빈도와 색상 매핑에 따라 워드클라우드를 생성하고 Base64로 반환합니다.
        """
        if not word_count:  # 빈 단어 리스트가 전달되면 워드클라우드 생성하지 않음
            raise ValueError("단어 리스트가 비어 있어 워드클라우드를 생성할 수 없습니다.")

        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            # 단어에 맞는 색상 반환 함수
            return word_color_mapping.get(word, 'black')  # 색상 기본값은 검정색

        # 워드클라우드 생성 (빈도수를 기반으로 워드클라우드를 만듦)
        wordcloud = WordCloud(
            font_path=font_path,
            width=600,
            height=300,
            background_color="white",
            prefer_horizontal=1,  # 가로로 배치될 확률을 높임
            max_font_size=50,  # 글자 크기의 최대값 설정
            font_step=2,  # 글자 크기 단계 설정
            relative_scaling=0.5,  # 단어 크기 조정
            color_func=color_func,  # 색상 지정 함수 추가
            scale=1,  # 전체 단어 크기 스케일
            max_words=25,  # 상위 max_words개 단어만 사용
            margin=10, ).generate_from_frequencies(word_count)

        # Base64로 인코딩
        buffer = BytesIO()
        wordcloud.to_image().save(buffer, format='PNG')
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return base64_image

    # 빈도수 기반으로 워드클라우드 생성
    wordcloud_base64 = save_wordcloud_base64(word_counts, word_color_mapping)

    def save_pie_chart_base64(sentiment_ratios):
        """
        감정 비율을 기반으로 도넛 모양의 파이차트를 생성하고 Base64로 반환합니다.
        """
        labels = [label.replace('\n', ' ') for label in sentiment_ratios.keys()]  # 멀티라인 텍스트 제거
        sizes = list(sentiment_ratios.values())  # 각 감정의 비율
        colors = ['#5745e9', '#ffc75a', '#ff6f6f']  # 각 감정의 색상 (긍정, 중립, 부정)
        explode = (0.0, 0, 0)  # 긍정적인 감정을 강조

        # 도넛 모양 파이차트 생성
        plt.figure(figsize=(6, 6))
        wedges, texts, autotexts = plt.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=140,
            wedgeprops={'width': 0.5}  # 도넛 모양으로 만드는 설정
        )

        # 텍스트 스타일 설정
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')

        plt.axis('equal')  # 원 형태로 만듦

        # Base64로 인코딩
        buffer = BytesIO()
        plt.savefig(buffer, format='PNG', bbox_inches='tight')
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        return base64_image

    # 감정별 비율 계산
    total_count = sum([len(words) for words in sentiment_results.values()])  # 총 단어 수
    sentiment_ratios = {sentiment: len(words) / total_count for sentiment, words in sentiment_results.items()}

    # 파이차트 생성
    pie_chart_base64 = save_pie_chart_base64(sentiment_ratios)

    # Base64 데이터 반환
    return wordcloud_base64, pie_chart_base64


def generate_tfidf_sentiment_visualizations(comments):
    # 1. 댓글 전처리
    try:
        print("댓글 전처리 시작...")
        processed_comments = [preprocess_text(comment) for comment in comments]
        if not processed_comments:
            raise ValueError("처리된 댓글이 없습니다. 입력된 댓글을 확인해 주세요.")
        print(f"처리된 댓글의 개수: {len(processed_comments)}개\n")
        print(f"처리된 댓글들: {processed_comments[:5]}...")  # 전처리된 댓글 출력 (상위 5개)
    except Exception as e:
        print(f"댓글 전처리 오류: {e}")
        return "댓글 전처리 오류"

    # 2. 명사와 동사만 추출
    try:
        print("명사와 동사 추출 시작...")
        extracted_words = []
        for comment in processed_comments:
            nouns_and_verbs = extract_nouns_and_verbs(comment)
            extracted_words.append(" ".join(nouns_and_verbs))
        print(f"명사와 동사 추출 완료: {extracted_words[:5]}...")
    except Exception as e:
        print(f"명사 및 동사 추출 오류: {e}")
        return "명사 및 동사 추출 오류"

    # 3. TF-IDF 분석을 통한 중요한 단어 추출
    try:
        print("TF-IDF 분석 시작...")
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(extracted_words)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        word_score_pairs = list(zip(feature_names, tfidf_scores))
        print(f"TF-IDF 분석 완료: {word_score_pairs[:5]}...")
    except Exception as e:
        print(f"TF-IDF 분석 오류: {e}")
        return "TF-IDF 분석 오류"

    # 4. TF-IDF 값 내림차순으로 정렬
    sorted_word_scores = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)

    # 5. 상위 100개 단어만 추출
    top_100_words = sorted_word_scores[:100]
    top_100_words_dict = {word: score for word, score in top_100_words}

    # 6. 각 단어의 빈도 계산
    word_counts = Counter()
    for comment in extracted_words:
        for word in comment.split():
            if word in top_100_words_dict:
                word_counts[word] += 1

    # 7. 빈도가 높은 상위 10개 단어 추출
    top_10_words = word_counts.most_common(10)
    print(f"빈도가 높은 상위 10개 단어: {top_10_words}\n")

    # 8. 감정 분석 결과 수집
    sentiment_results = []

    # 9. 각 단어에 대해 감정 분석 수행
    for rank, (word, frequency) in enumerate(top_10_words, start=1):
        print(f"순위 {rank}, 단어: {word}, 빈도: {frequency}에 대해 감정 분석 중...")
        sentiment = analyze_sentiment(word)
        sentiment_results.append((rank, word, sentiment))  # 순위 추가

    # 10. 결과를 데이터프레임으로 변환
    sentiment_df = pd.DataFrame(sentiment_results, columns=["순위", "단어", "감정"])

    # 11. HTML 스타일 추가
    custom_style = """
    <style>
    .table {
        table-layout: auto;
        width: auto;
        max-width: 100%;
    }
    .table th, .table td {
        padding: 5px; /* 셀 내부 여백 최소화 */
        text-align: center; /* 텍스트 중앙 정렬 */
        white-space: nowrap; /* 셀 크기 단어 크기에 맞춤 */
    }
    </style>
    """

    # 12. HTML로 변환하여 반환
    sentiment_html = custom_style + sentiment_df.to_html(index=False, classes="table table-bordered table-striped")
    return sentiment_html


def main():
    # 예시 댓글 데이터 (이것은 실제 데이터로 대체해야 합니다)
    comments = [
        {'author': '@dk-fo5wh', 'comment': '이재명잡아 들여 재판열어라~너네가 오죽했으면 대통령께서 비상계엄을 내렸겠나!!'},
        {'author': '@user1', 'comment': '정말 비상계엄을 내리다니... 이게 뭐야?'},
        {'author': '@user2', 'comment': '대통령이 그런 결정을 했다니 충격이다.'},
        # 더 많은 댓글을 추가하세요...
    ]

    # 댓글에서 텍스트만 추출하여 리스트로 만듦
    comment_texts = [comment['comment'] for comment in comments]

    # TF-IDF 분석을 통해 중요한 단어 및 감정 분석 결과 얻기
    result_html_table = save_visualizations_with_tfidf(comment_texts)

    # 결과를 출력 (또는 HTML 파일로 저장 등)
    print(result_html_table)