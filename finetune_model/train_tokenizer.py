from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "gpt2"               # 124M parameters
# MODEL="facebook/xglm-564M"

LANGUAGE = "amh"
DATASET_PATH = f'datasets/{LANGUAGE}_dedup_filtered/*.arrow'
TOKENIZER_OUTPUT_PATH = f'models/tokenizer/{MODEL}-{LANGUAGE}-tokenizer'

raw_datasets = load_dataset("arrow", data_files={DATASET_PATH})

'''Assembling a corpus'''

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["text"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )
training_corpus = get_training_corpus()

# Don't uncomment the following line unless your dataset is small!
# training_corpus = [raw_datasets["train"][i: i + 1000]["text"] for i in range(0, len(raw_datasets["train"]), 1000)]

'''Training a new tokenizer'''  
old_tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 50257)
tokenizer.save_pretrained(TOKENIZER_OUTPUT_PATH)

example = "ትክክለኛው ዋጋ ካምፓኒው ጊዜ የተወሰነ ነጥብ ላይ አንድ የንግድ ኦፕሬሽን ለመምራት ዝግጁ ነው ላይ የተጠቀሰ ነው."
tokens = tokenizer.tokenize(example)
print(tokens)
print()
