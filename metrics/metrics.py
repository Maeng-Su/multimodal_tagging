import ast
import pandas as pd
import numpy as np
import fasttext
import torch
from sentence_transformers import util
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')


# 정답 CSV의 Hashtag 컬럼 처리: 콤마로 구분된 문자열을 리스트로 변환
def process_gt_hashtag(hashtag_str):
    if pd.isna(hashtag_str):
        return []
    return [x.strip() for x in hashtag_str.split(',') if x.strip()]

# 예측 CSV의 generated_tags 컬럼 처리: 문자열 형태의 리스트를 ast.literal_eval로 파싱
def process_predicted_tags(tag_str):
    try:
        if isinstance(tag_str, str) and tag_str.startswith('['):
            tags = ast.literal_eval(tag_str)
            if isinstance(tags, list):
                return [str(t).strip() for t in tags]
        return [x.strip() for x in tag_str.split(',') if x.strip()]
    except Exception as e:
        return []

# FastText 기반 평가 지표: 각 이미지에 대해, top k 예측 태그와 ground truth 태그 간
# 헝가안 알고리즘을 통해 최적 매칭 후 threshold 이상의 매칭을 올바른 것으로 간주하여 계산
def fasttext_single_metrics(truth, predict, model_ft, k=8, threshold=0.6):
    pred_tags = predict[:k]
    if not pred_tags or not truth:
        return 0, 0, 0

    # 각 태그의 임베딩 계산
    pred_embeddings = [model_ft.get_word_vector(tag) for tag in pred_tags]
    truth_embeddings = [model_ft.get_word_vector(tag) for tag in truth]

    pred_tensor = torch.tensor(pred_embeddings)
    truth_tensor = torch.tensor(truth_embeddings)
    
    # 코사인 유사도 매트릭스 계산 (행: truth, 열: pred)
    sim_matrix = util.pytorch_cos_sim(truth_tensor, pred_tensor)
    cost = -sim_matrix.cpu().numpy()  # 최대 유사도를 위해 음수로 변환
    
    # 헝가안 알고리즘을 이용한 최적 매칭
    row_idx, col_idx = linear_sum_assignment(cost)
    
    # 매칭된 태그 쌍 중 threshold 이상의 유사도를 올바른 매칭으로 간주
    match_count = 0
    for i, j in zip(row_idx, col_idx):
        if sim_matrix[i, j].item() >= threshold:
            match_count += 1

    precision = match_count / len(pred_tags)
    recall = match_count / len(truth)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def main():
    # 파일 경로 지정
    gt_file = "/home/downtown/aiffel/miniDLThon/dataset_for_evaluate1.csv"
    pred_file = "/home/downtown/aiffel/miniDLThon/gemma3_generated_tags_eval1_4.csv"
    
    # CSV 파일 읽기
    df_gt = pd.read_csv(gt_file)
    df_pred = pd.read_csv(pred_file)
    
    # CSV 컬럼 이름 확인
    print("정답 CSV 컬럼:", df_gt.columns.tolist())
    print("예측 CSV 컬럼:", df_pred.columns.tolist())
    
    # 두 DataFrame을 "Image Path"를 기준으로 병합
    df_merged = pd.merge(df_gt, df_pred, on="Image Path", how="inner", suffixes=("_gt", "_pred"))
    
    # 정답 태그와 예측 태그 처리
    df_merged["gt_tags"] = df_merged["Hashtag"].apply(process_gt_hashtag)
    df_merged["pred_tags"] = df_merged["generated_tags"].apply(process_predicted_tags)
    
    # 평가 지표 계산을 위해 리스트로 변환
    truth_list = df_merged["gt_tags"].tolist()
    predict_list = df_merged["pred_tags"].tolist()
    
    # FastText 모델 로드
    model_ft = fasttext.load_model('cc.ko.300.bin')
    
    # 각 샘플별 FastText 기반 평가 지표 계산 (top 8 예측 태그 기준)
    precisions = []
    recalls = []
    f1s = []
    
    for truth, pred in zip(truth_list, predict_list):
        p, r, f = fasttext_single_metrics(truth, pred, model_ft, k=8, threshold=0.6)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    
    avg_precision = np.average(precisions)
    avg_recall = np.average(recalls)
    avg_f1 = np.average(f1s)
    
    print("FastText Precision:", avg_precision)
    print("FastText Recall:", avg_recall)
    print("FastText F1 Score:", avg_f1)
    
    # 평가 결과를 병합된 DataFrame에 추가
    df_merged["fasttext_precision"] = precisions
    df_merged["fasttext_recall"] = recalls
    df_merged["fasttext_f1"] = f1s
    
    # 저장할 최종 컬럼만 선택
    cols_to_save = ["Image Path", "Text_gt", "Hashtag", "generated_tags", 
                    "fasttext_precision", "fasttext_recall", "fasttext_f1"]
    df_final = df_merged[cols_to_save]
    
    # CSV로 저장
    output_file = "/home/downtown/aiffel/miniDLThon/evaluation_results_fasttext.csv"
    df_final.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("평가 결과 CSV가 저장되었습니다:", output_file)

if __name__ == "__main__":
    main()
