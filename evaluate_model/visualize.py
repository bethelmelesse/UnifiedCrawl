import numpy as np
import pandas as pd
from hyperparameters import DEBUG, MODEL, LANGUAGE
import os


def print_output_and_scores(questions, ground_truth, all_predictions, max_f1_score, max_em_score):

    # TODO: join BASE_OUTPUT_PATH and path using os.path.join
    if "models" in MODEL:
        BASE_OUTPUT_PATH = f'{MODEL}/evaluation/'
    else:
        BASE_OUTPUT_PATH = f'models/original_output/{LANGUAGE}/{MODEL}/evaluation/'
    if DEBUG:
        file_name = BASE_OUTPUT_PATH + "debug_output.xlsx"          # for visiualizing 
        file_name_scores = BASE_OUTPUT_PATH + "debug_scores.xlsx"   # for recording final score when debug is false
    else:
        file_name = BASE_OUTPUT_PATH + "final_output.xlsx"          # for visiualizing 
        file_name_scores = BASE_OUTPUT_PATH + "final_scores.xlsx"   # for recording final score when debug is false

    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)

    d = {"Questions" : questions, "Ground Truth" : ground_truth, "All Predictions" : all_predictions, "max_f1_score": max_f1_score, "max_em_score" : max_em_score}
    df = pd.DataFrame(data=d)
    
    df.to_excel(file_name)

    mean = [np.mean(max_f1_score), np.mean(max_em_score)]
    score_type = ["max_f1_score", "max_em_score"]
    d1 = {"Score Type" : score_type, "Mean" : mean}
    df1 = pd.DataFrame(data=d1)
    print(df1.to_string(index=False))
    df1.to_excel(file_name_scores)
