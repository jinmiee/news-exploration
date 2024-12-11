# -*- coding: utf-8 -*-
import sys
import io

# 표준 출력 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from bareunpy import Tagger

# 아래에 "https://bareun.ai/"에서 이메일 인증 후 발급받은 API KEY("koba-...")를 입력해주세요. "로그인-내정보 확인"
API_KEY="koba-5JNWNQI-MH5EQJY-QINVNIQ-2IOA5IY" # <- 본인의 API KEY로 교체 

# 방금 설치한 자신의 호스트에 접속합니다.
tagger = Tagger(API_KEY, 'localhost')
# 결과를 가져옵니다.
res = tagger.tags(["안녕하세요.", "바른을 사용해서 새로운 경험을 해보세요."])
# 용어만 뽑아냅니다.
va = res.verbs()
print(va)
