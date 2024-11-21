from datasets import load_dataset
from tqdm import tqdm
from generate_prompt import choose_dataset, make_prompt
from hyperparameters import LANGUAGE, NUM_PROMPT_EXAMPLE, BATCH_SIZE, DEBUG, device, random, MODEL, MAX_LENGTH
from utils import stopping_criteria_fn

def get_prediciton_and_ground_truth(model, tokenizer):      # dataset_path = amhqa
    dataset_path, prompt_details = choose_dataset(dataset_lang_type=LANGUAGE)
    my_dataset = load_dataset(dataset_path)
    prompt_example_index = random.sample(range(len(my_dataset["train"])), NUM_PROMPT_EXAMPLE)
    prompt_context = ""
    for i in prompt_example_index:
        prompt_context += make_prompt(i, prompt_details=prompt_details, my_dataset=my_dataset,
                                      dataset_type="train", include_answer=True, tokenizer=tokenizer) 
    all_predictions = []
    ground_truth = []
    questions = []

    stopping_criteria=None
    if 'llama' in MODEL or 'bloom' in MODEL:
        stopping_criteria = stopping_criteria_fn(tokenizer)

    total_steps = int(len(my_dataset["validation"])/BATCH_SIZE)
    if DEBUG:
        total_steps = 10
    steps = random.sample(range(int(len(my_dataset["validation"])/BATCH_SIZE)), total_steps)
    # steps =range(10)

    for step in tqdm(steps):
        # Warning: DO NOT USE STEP FOR ANYTHING OTHER THAN EXAMPLES. BECAUSE IT WILL BE RANDOM.
        prompt_storage = []
        for current_example in range(BATCH_SIZE):
            next_example = current_example + BATCH_SIZE * step
            prompt = prompt_context + make_prompt(next_example, prompt_details, my_dataset, dataset_type="validation", include_answer=False, tokenizer=tokenizer)
            prompt_storage.append(prompt)
            ground_truth.append(my_dataset["validation"][next_example]["answers"]["text"])
            questions.append(my_dataset["validation"][next_example]['question'])
        
        inputs = tokenizer(prompt_storage, return_tensors="pt", padding='longest', truncation=True, max_length=MAX_LENGTH).input_ids.to(device) 
        # if inputs.shape[1] > 2000:
        #     size_to_be_removed = inputs.shape[1] - 2000
        #     inputs = inputs[:, size_to_be_removed:]

        
        # Use this if doing few shot prompting inference
        outputs = model.generate(inputs=inputs, max_new_tokens=40, do_sample=True, top_k=4, top_p=0.95, stopping_criteria=stopping_criteria)

        # Use this if doing supervised inference
        # outputs = model.generate(inputs=inputs, max_new_tokens=20, do_sample=False, num_beams=5, stopping_criteria=stopping_criteria)

        predictions = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)
        predictions = [prediction.split('\n')[0] for prediction in predictions]
        predictions = [prediction.split(';')[0] for prediction in predictions]
        all_predictions += predictions
        
    return all_predictions, ground_truth, questions