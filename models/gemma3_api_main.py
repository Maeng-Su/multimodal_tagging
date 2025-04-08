import pickle
import random
import tensorflow as tf
import requests
import json
import base64
from PIL import Image
from io import BytesIO
import pandas as pd
import os
import re
import ast

'''
현재 별도로 구성된 평가 데이터셋(csv file)을 사용하는 것을 기준으로 작성된 코드입니다.
원본 kts의 pickle 파일을 이용한 추론 시, '# "kts/test.pickle" inference 시 사용' 주석이 달린 부분을 활성화하고,
기존에 사용되던 부분을 주석처리 하여 코드를 실행하시면 됩니다.

코드 실행 절차
1. Ollama 설치 # window, linux, mac 환경에 따라 설치 방법 상이
2. $ Ollama run gemma3:4b # 모델 불러오기
3. $ python gemma3_api_main.py # 코드 실행 
'''

'''
# "kts/test.pickle" inference 시 사용

class KTSDataset():
    def __init__(self, data_address="kts", base_dir="/home/downtown/aiffel/miniDLThon/Korean-Tourist-Spot-Dataset-master", random_seed=-1):
        self.database_address = data_address
        self.base_dir = base_dir  # 이미지가 저장된 기본 디렉토리
        self.random_seed = random_seed
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)
        
    def _get_parse_function(self):
        def parse_function(img_path):
            image = tf.image.decode_jpeg(tf.io.read_file(img_path))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)
            return image / 255.
        return lambda img_path: tf.py_function(parse_function, [img_path], tf.float32)
    
    def make_dataset(self, mode):
        images = []
        tags = []
        texts = []
        self.num_data = 0
        pickle_path = os.path.join(self.base_dir, "kts/test.pickle")

        with open(pickle_path, "rb") as fr:
            dataset = pickle.load(fr)

        for data in dataset:
            # 원래 경로에서 백슬래시를 슬래시로 변경
            data['img_name'] = data['img_name'].replace('\\', '/')

            # "amusement park"가 경로에 포함된 경우 건너뛰기
            if "amusement park" in data['img_name']:
                print(f"건너뛰는 이미지 (amusement park): {data['img_name']}")
                continue

            print("데이터:", data)
            if data["hashtag"] != []:
                full_img_path = os.path.join(self.base_dir, data["img_name"])
                images.append(data["img_name"])  # CSV에 저장할 때는 원본 경로 사용
                tags.append(random.choice(data["hashtag"]))
                texts.append(data.get("text", ""))
                self.num_data += 1

        return images, tags, texts
        '''

# "dataset_for_evaluate1.csv" inference 시 사용

class KTSDataset():
    def __init__(self, csv_path, random_seed=-1):
        self.csv_path = csv_path
        self.random_seed = random_seed
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)
        
    def make_dataset(self):
        images = []
        texts = []
        # CSV 파일을 pandas로 읽어옴
        df = pd.read_csv(self.csv_path)
        
        # CSV 파일에 "Image Path", "Text" 열이 있다고 가정
        for idx, row in df.iterrows():
            image_path = row["Image Path"].replace('\\','/')
            text = row["Text"]
            
            # 필요 시 "Hashtag" 열도 사용 가능
            # if "Hashtag" in df.columns:
            #    hashtag = row["Hashtag"]
            
            # # 필요하다면 "amusement park" 등 특정 경로 제외 처리
            # if "amusement park" in image_path:
            #     print(f"건너뛰는 이미지 (amusement park): {image_path}")
            #     continue
            
            images.append(image_path)
            texts.append(text)
        
        # 해시태그 등은 사용하지 않는다면 반환하지 않아도 됨
        return images, texts


def image_to_base64(full_img_path):
    try:
        with Image.open(full_img_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"이미지 {full_img_path} 열기 오류: {e}")
        return None

def parse_tags(response_str):
    """
    모델 응답 문자열에서 JSON 배열 또는 불릿(*) 형태 태그 리스트를 추출합니다.
    1. 대괄호([]) 안의 JSON 배열 형식 우선 시도
    2. 불릿(*) 형식 라인들을 추출
    3. 괄호() 안의 설명은 제거하고, 영어 괄호 내용은 별도 태그로 처리
    4. 최종적으로 최대 8개의 태그만 반환
    """
    final_tags = []
    
    # 1) 대괄호 형태 우선
    bracket_pattern = re.compile(r"(\[.*?\])", re.DOTALL)
    bracket_match = bracket_pattern.search(response_str)
    if bracket_match:
        try:
            bracket_list = ast.literal_eval(bracket_match.group(1))
            if isinstance(bracket_list, list):
                final_tags.extend([str(tag).strip() for tag in bracket_list])
        except Exception as e:
            print("대괄호 파싱 오류:", e)
    
    # 2) 불릿(*) 라인 형태
    bullet_pattern = re.compile(r"^\*\s+(.*)$", re.MULTILINE)
    bullet_lines = bullet_pattern.findall(response_str)
    if bullet_lines:
        final_tags.extend([line.strip() for line in bullet_lines])
    
    # 3) 추가 후처리: 괄호 제거 및 쉼표, 공백 분할
    processed = []
    for tag in final_tags:
        # 영어 괄호 내용 추출 (예: "멜팅 메로우(MeltingNouNou)" → 별도 태그)
        paren_eng = re.findall(r"\(([A-Za-z0-9]+)\)", tag)
        tag_no_paren = re.sub(r"\(.*?\)", "", tag)  # 괄호 전체 제거
        # 쉼표 또는 공백으로 분할 (하나의 태그 내에 공백이 있으면 분리)
        parts = re.split(r"[,\s]+", tag_no_paren)
        parts = [part.strip() for part in parts if part.strip()]
        processed.extend(parts)
        processed.extend(paren_eng)
    
    # 중복 제거 및 순서 유지 (필요하다면)
    seen = set()
    unique_tags = []
    for t in processed:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)
    
    # 최대 8개만 반환
    return unique_tags[:8]

def get_tags_from_gemma(image_path, text, base_dir, model_name="gemma3:latest"):
    api_url = "http://localhost:11434/api/generate" # 앤드포인트
    
    full_img_path = os.path.join(base_dir, image_path)

    image_b64 = image_to_base64(full_img_path)
    if image_b64 is None:
        return None
    
    # 엄격한 프롬프트: 오직 JSON 배열만 반환하도록 요구합니다.
    # 태그 생성 결과 확인 후 상황에 따라 조금씩 prompt를 수정해도 좋습니다.

    # gemma3_generated_tags_eval1.csv

    # prompt = (
    # "당신은 엄격한 JSON 생성기입니다. "
    # "오직 8개의 문자열 태그로 구성된 JSON 배열만 응답하세요. "
    # "추가적인 설명이나 문장은 절대 포함하지 말고, 가능한 한국어 태그만 사용하세요. "
    # "예시: [\"태그1\", \"태그2\", \"태그3\", \"태그4\", \"태그5\", \"태그6\", \"태그7\", \"태그8\"]\n\n"
    # f"참고 텍스트: {text}\n"
    # )

    # gemma3_generated_tags_eval1_2.csv

    # prompt = (
    # "당신은 엄격한 JSON 생성기입니다. "
    # "오직 8개의 순수한 한국어 문자열 태그로 구성된 JSON 배열만 응답하세요. "
    # "추가적인 설명이나 문장은 절대 포함하지 말고, 한국어 태그만 사용하세요. "
    # "다음 형식을 지켜주세요 : [\"태그1\", \"태그2\", \"태그3\", \"태그4\", \"태그5\", \"태그6\", \"태그7\", \"태그8\"] "
    # f"참고 텍스트: {text}\n"
    # )

    # gemma3_generated_tags_eval1_3.csv    

    # prompt = (
    # "당신은 엄격한 JSON 생성기입니다. "
    # "오직 8개의 순수한 한국어 문자열 태그로 구성된 JSON 배열만 응답하세요. "
    # "다음 형식을 지켜주세요 : [\"태그1\", \"태그2\", \"태그3\", \"태그4\", \"태그5\", \"태그6\", \"태그7\", \"태그8\"] "
    # f"참고 텍스트: {text}\n"
    # )

    # gemma3_generated_tags_eval1.csv    

    prompt = (
    "당신은 엄격한 JSON 생성기입니다. "
    "반드시 다음 형식을 지켜 8개의 한국어 태그를 생성해주세요 : [\"태그1\", \"태그2\", \"태그3\", \"태그4\", \"태그5\", \"태그6\", \"태그7\", \"태그8\"] "
    f"참고 텍스트: {text}\n"
    )

    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_b64],
        "format": "json",
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        print("=== Raw response text ===")
        print(response.text)

        try:
            response_json = response.json()
        except Exception as e:
            print("JSON 파싱 에러:", e)
            return None

        if response.status_code == 200:
            print(f"API 응답 (파일: {image_path}):", response_json)
            raw_resp = response_json.get("response", "")
            if not raw_resp:
                print("응답에 'response' 키가 없음:", response_json)
                return None

            # 후처리: 문자열에서 태그 리스트만 추출
            tag_list = parse_tags(raw_resp)
            return tag_list
        else:
            print(f"API 호출 실패 (상태코드 {response.status_code}): {response.text}")
            return None

    except Exception as e:
        print(f"API 호출 오류: {e}")
        return None

'''

# "kts/test.pickle" inference 시 사용

def main():
    base_dir = "/home/downtown/aiffel/miniDLThon/Korean-Tourist-Spot-Dataset-master"

    dataset_instance = KTSDataset("kts")
    # relative_paths: 피클에 있는 원본 경로 (예: "kts/images/tower/997.jpg")
    relative_paths, orig_tags, texts = dataset_instance.make_dataset("test")
    
    results = []  # 결과 저장 리스트
    for rel_path, text in zip(relative_paths, texts):
        print(f"처리 중: {rel_path}")
        tags_generated = get_tags_from_gemma(rel_path, text)
        results.append({
            "image_path": rel_path,   # 피클 파일의 원본 경로 그대로 저장
            "text": text,             # 이미지와 함께 사용된 텍스트
            "generated_tags": tags_generated  # 후처리로 추출한 태그 리스트
        })
    
    df = pd.DataFrame(results)
    csv_filename = "gemma3_generated_tags_test.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"CSV 파일이 저장되었습니다: {csv_filename}")

if __name__ == "__main__":
    main()
'''

# "dataset_for_evaluate1.csv" inference 시 사용

def main():
    # CSV 파일 경로와 base_dir 지정
    csv_path = "/home/downtown/aiffel/miniDLThon/dataset_for_evaluate1.csv"
    base_dir = "/home/downtown/aiffel/miniDLThon/Korean-Tourist-Spot-Dataset-master"
    
    dataset_instance = KTSDataset(csv_path)
    image_paths, texts = dataset_instance.make_dataset()
    
    results = []
    for (rel_path, text) in zip(image_paths, texts):
        print(f"처리 중: {rel_path}")
        tags_generated = get_tags_from_gemma(rel_path, text, base_dir)
        results.append({
            "image_path": rel_path,
            "text": text,
            "generated_tags": tags_generated
        })
    
    df = pd.DataFrame(results)
    csv_filename = "gemma3_generated_tags_eval1.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"CSV 파일이 저장되었습니다: {csv_filename}")

if __name__ == "__main__":
    main()
