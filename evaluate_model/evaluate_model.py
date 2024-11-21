from evaluate import load
from tqdm import tqdm

def evaluate_model(all_predictions, ground_truth):
    metric = load("squad")
    max_f1_score = []
    max_em_score = []

    for i in tqdm(range(len(ground_truth))):
        predictions = [{'prediction_text': all_predictions[i], 'id': '56e10a3be3433e1400422b22'}]
        score_for_each_ans_in_ground_truth = []

        for j in range(len(ground_truth[i])):
            references = [{'answers': {'answer_start': [97], 'text': [ground_truth[i][j]]}, 'id': '56e10a3be3433e1400422b22'}]
            score = metric.compute(predictions=predictions, references=references)
            score_for_each_ans_in_ground_truth.append(score)
            
        # print(f"{i} = {score_for_each_ans_in_ground_truth}")
        score_f1 = [item['f1'] for item in score_for_each_ans_in_ground_truth]
        score_em = [item['exact_match'] for item in score_for_each_ans_in_ground_truth]
       
        max_f1_score.append(max(score_f1))
        max_em_score.append(max(score_em))

    return max_f1_score, max_em_score   