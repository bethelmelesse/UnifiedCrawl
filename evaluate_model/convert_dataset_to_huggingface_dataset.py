from datasets import load_dataset
import json
import os

output_directory = "amh_txt_collections_hf"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# input_directory = "amh_txt_collections/amh_txt_col0.json"
input_directory = "amh_txt_collections"

dir_list = os.listdir(input_directory)

for file in dir_list:
    new_dataset_path = os.path.join(output_directory, file)
    read_path = os.path.join(input_directory, file)

    with open(read_path) as f:
        read_file = json.load(f)

    list_of_dic = []
    for i in range(len(read_file)):
        dic = {"text": f"{read_file[i]}"}
        list_of_dic.append(dic)

    with open(new_dataset_path, "w", encoding='utf-8') as outfile:
        json.dump(list_of_dic, outfile, ensure_ascii=False)


dataset = load_dataset('json', data_files=f"{output_directory}/*.json")
print()