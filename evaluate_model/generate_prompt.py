from hyperparameters import MODEL

def choose_dataset(dataset_lang_type):
    if dataset_lang_type == "amh":
        dataset_path = "evaluate_model/amhqa"
        passage_prompt = "አንቀጽ:"
        questions_prompt = "ጥያቄ:"
        answers_prompt = "መልስ:"

    elif dataset_lang_type == "en":
        # dataset_path  = "/home/bethel/git/amh_data_collection/squad"
        dataset_path = "squad"
        passage_prompt = "PASSAGE:"
        questions_prompt = "QUESTION:"
        answers_prompt = "ANSWER:"
    prompt_details = [passage_prompt, questions_prompt, answers_prompt]
    return dataset_path, prompt_details

def make_prompt(i, prompt_details, my_dataset, dataset_type, include_answer, tokenizer):
    context = my_dataset[dataset_type][i]['context']
    question = my_dataset[dataset_type][i]['question']

    joiner = '\n'
    if "xglm" in MODEL:
        joiner = '; '
    
    if include_answer == True:
        if "xglm" in MODEL:
            answer = my_dataset[dataset_type][i]['answers']['text'][0]  + tokenizer.eos_token
        elif "gpt2" in MODEL or "llama" in MODEL or "bloom" in MODEL:
            answer = my_dataset[dataset_type][i]['answers']['text'][0] + "\n"

    else:
        answer = ""

    passage_prompt = prompt_details[0]
    questions_prompt = prompt_details[1]
    answers_prompt = prompt_details[2]
    # TO DO -- change \n
    prompt_context = passage_prompt + context + joiner + questions_prompt + question + joiner + answers_prompt + answer
    return prompt_context




