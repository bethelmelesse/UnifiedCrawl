import json
import os
from natsort import natsorted
from tqdm import tqdm
from glob import glob

LANGUAGE='amh'

BASE_PATH=f'datasets/CC-MAIN-*/{LANGUAGE}_txt_collections_hf'
# OUT_PATH=f'datasets/{LANGUAGE}_txt_combined.jsonl'

cc_crawls = natsorted(glob(BASE_PATH))
print(f'Joining these crawls - {cc_crawls}')

# files = [file for file in files if int(file.split('.')[0])<9]
def process_single_crawl(cc_crawl):
    out_path = cc_crawl + '_combined.jsonl'
    with open(out_path, 'w', encoding='utf-8') as fout:
        files = natsorted(os.listdir(cc_crawl))
        # Across all files in one crawl
        for file in files:
            with open(os.path.join(cc_crawl, file)) as fin:
                input_data = json.load(fin)

                # Go through each "document" (web-page) in each file
                for document in input_data:
                    if document['text'] is None:    # ignore none document because our text filtering sometimes gives none document
                        continue
                    output_line = json.dumps(document, ensure_ascii=False) + '\n'
                    fout.write(output_line)

# across different crawls
for cc_crawl in tqdm(cc_crawls):
    print(f'currently doing crawl {cc_crawl}')
    process_single_crawl(cc_crawl)

    