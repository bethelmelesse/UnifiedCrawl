import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from hyperparameters import device, MODEL

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop):
        super().__init__()
        self.stop = stop

    def __call__(self, input_ids, _):
        for example_input_ids in input_ids:
            if torch.all((self.stop == example_input_ids[-len(self.stop):])).item():
                return True

        return False

def stopping_criteria_fn(tokenizer):
    # stop_words = ["PASSAGE:","QUESTION:", "ANSWER:", "\n"]
    # stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    if 'llama' in MODEL:
        stop_words_ids = torch.tensor([13]).to(device)
    elif 'bloom' in MODEL:
        stop_words_ids = torch.tensor([189]).to(device)

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_ids)])
    return stopping_criteria
