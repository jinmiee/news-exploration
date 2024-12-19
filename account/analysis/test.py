'''
# KPF-BERT 적용 (토크나이저까지만 됨)
from transformers import BertModel, BertTokenizer
import sys

# UTF-8 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

model_name_or_path = r"C:\siibal"

model = BertModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
tokenizer = BertTokenizer.from_pretrained(
    model_name_or_path, 
    local_files_only=True,
    encoding='utf-8'
)

text = "언론진흥재단 BERT 모델을 공개합니다."
tokens = tokenizer.tokenize(text)
print(tokens)

encoded = tokenizer(text)
print(encoded)
'''








from transformers import AutoModelForMaskedLM, AutoTokenizer, logging
import torch
import sys

# UTF-8 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

# 로깅 레벨 설정
logging.set_verbosity_error()

# KPF-BERT 모델 경로 또는 이름
model_name_or_path = r"C:\siibal"

# 모델과 토크나이저 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, local_files_only=True)
    model.eval()
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    raise

def correct_text(text, max_chunk_length=128):
    """
    텍스트의 문법 및 철자 오류를 교정하는 함수
    """
    try:
        # 텍스트를 적절한 크기로 분할
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_chunk_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        corrected_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"청크 {i+1}/{len(chunks)} 처리 중...")
            
            # 토큰화
            inputs = tokenizer(chunk, 
                               return_tensors="pt", 
                               max_length=512, 
                               truncation=True,
                               padding=True)
            
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]
            corrected_ids = input_ids.clone()
            
            # 각 토큰에 대해 마스킹 및 예측 수행
            for j in range(1, len(input_ids) - 1):
                if attention_mask[j] == 0:
                    continue
                    
                masked_input_ids = input_ids.clone()
                masked_input_ids[j] = tokenizer.mask_token_id
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=masked_input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0)
                    )
                
                predictions = outputs.logits[0, j]
                probs = torch.softmax(predictions, dim=-1)
                predicted_id = torch.argmax(predictions)
                max_prob = probs[predicted_id]
                
                # 높은 확률로 예측된 경우에만 수정
                if max_prob > 0.9 and predicted_id != input_ids[j]:
                    corrected_ids[j] = predicted_id
            
            # 디코딩
            corrected_chunk = tokenizer.decode(corrected_ids, skip_special_tokens=True)
            corrected_chunks.append(corrected_chunk)
        
        return ' '.join(corrected_chunks)

    except Exception as e:
        print(f"텍스트 교정 중 오류 발생: {e}")
        return text

def main():
    text = """윤성열 대통령이 머물고 있는 한남동 관절로 가보겠습니다 자 정인나 기자 윤성열 대통령 오늘 오전 공수처에 출석 했어야 하는데 결국 나타나지 않았습니다 관저에 계속 머물고 있는 건가요네 윤 대통령은 지난 12일 대국민 담화를 발표한 이후 지금까지 모습을 드러내지 않고 있습니다 제가 아침부터 지금까지 계속 지켜봤는데요 대통령과 경호차량 행렬을 볼 수 없었습니다 오전 10시 10분 우체국 오토바이가 관저 앞으로 와서 소포를 전달하는 모습이 포착됐습니다 이 소포의 정체가 뭔지 관저 안으로 전달됐는지 확인되지 않았습니다 어제도 우체국 차량이 관저 앞으로 와서 공조 본내 출석 요구서를 전달했"""
    
    try:
        print("텍스트 교정 시작...")
        corrected_text = correct_text(text)
        
        print("\n=== 원본 텍스트 ===")
        print(text[:200])
        print("\n=== 교정된 텍스트 ===")
        print(corrected_text[:200])
        
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
