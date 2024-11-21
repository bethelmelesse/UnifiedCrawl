from transformers import AutoTokenizer
import re

MODEL="facebook/xglm-4.5B"

passage_prompt = "አንቀጽ:"
questions_prompt = "ጥያቄ:"
answers_prompt = "መልስ:"
joiner = '; '

context = 'ጠቅላይ ሚኒስትር ዐቢይ አሕመድ ከ2010 ጀምሮ በፋይናንሱ ዘርፍ ስኬታማ ለውጦች መመዝገባቸውን ገለጹ፡፡ ጠቅላይ ሚኒስትር ዐቢይ የፋይናንስ ዘርፍ ዐበይት ስኬቶች በሚል በማህበራዊ ትስስር ገፃቸው ላይ እንዳስታወቁት የታክስ ገቢ በ2010 ከነበረበት 229 ቢሊየን ብር በ2012 የ36 በመቶ ጭማሪ በማሳየት 311 ቢሊየን ማድረስ ተችሏል።'
question = "የታክስ ገቢ ከ2010-2012 በመቶኛ የምን ያህል መጠን እድገት አሳየ?"
answer = "የ36 በመቶ"

# answer_char_length = len(answer)

prompt_context = passage_prompt + context + joiner + questions_prompt + question + joiner + answers_prompt + answer


# match=(re.search(answer, prompt_context))
starting_index_of_ans = len(prompt_context) - len(answer)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
encoded = tokenizer(prompt_context, return_offsets_mapping=True)
input_ids = encoded['input_ids']
offset_mappings = encoded['offset_mapping']

# index_of_ans_beg = offset_mappings.index((starting_index_of_ans, ))

for i in range(len(offset_mappings)):
    # print(i, offset_mappings[i])
    a, b = offset_mappings[i]
    if a <= starting_index_of_ans < b: 
        starting_index_of_ans_token = i

print()