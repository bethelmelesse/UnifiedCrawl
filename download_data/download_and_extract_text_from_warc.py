import pandas as pd        # think of it as excel sheet
import glob
import boto3
from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
from botocore.config import Config
from natsort import natsorted
from trafilatura import fetch_url, extract
import os
import json
import multiprocessing
import argparse

s3_region='us-east-1'
s3_access_key_id='ADD_YOUR_ID'
s3_secret_access_key='ADD_YOUR_KEY'
config = Config(
retries = {
    'max_attempts': 10,
    'mode': 'adaptive'
}
)
client = boto3.client('s3',region_name = s3_region, aws_access_key_id=s3_access_key_id, aws_secret_access_key=s3_secret_access_key, config=config)


def read_input(path_to_files, language):
    df = pd.DataFrame()
    for f in path_to_files:
        csv = pd.read_csv(f)
        df = pd.concat([csv, df], ignore_index=True)        # 117191 rows * 5 columns
        # list(df)--- to see the header of the columns
    print()

    # check how many are in just LANGUAGE and how many are in LANGUAGE and eng
    # TODO: Check this line
    only_lang = (df[df.content_languages == f'{language}']).reset_index()
    return only_lang


def downloadwarc(row):
    # time.sleep(1)
    start = row.warc_record_offset
    end = start + row.warc_record_length -1
    r = client.get_object(Bucket='commoncrawl', Key=row.warc_filename, Range=f"bytes={start}-{end}")

    html = None
    for record in ArchiveIterator(r['Body']):
        if record.rec_type == 'response':
            html = record.content_stream().read()
    
    return html

def write_on_file(count_folders, text_list, path_to_output_files):
            file_name = f"{count_folders}.json"
            full_path = os.path.join(path_to_output_files, file_name)
            with open(full_path, "w", encoding='utf-8') as f:
                # Convert list[stringA, stringB,...] to list of dict [{"text": stringA}, {"text": stringB}]
                output_list = [{"text": x} for x in text_list]
                json.dump(output_list, f, ensure_ascii=False)

def check_last_folder(path_to_output_files):
    previously_done = natsorted(glob.glob(f"{path_to_output_files}*.json"))
    if len(previously_done) == 0:
        return -1
    
    previously_done = previously_done[-1]
    previously_done = previously_done.split('/')[-1]
    previously_done = int(previously_done[:-5])
    print('Starting from file number - ', previously_done)
    return previously_done

# This will happen inside multi-processing process
def download_and_extract(joint_input):
    index, row = joint_input
    html = downloadwarc(row)
    text_from_html = extract(html)
    return index, text_from_html


def main():
    parser = argparse.ArgumentParser()
    # python will change this to underscore
    parser.add_argument("--cc-crawl-version", default='CC-MAIN-2023-06', help="CC version to download, example CC-MAIN-2023-06")
    args = parser.parse_args()
    cc_crawl_version = args.cc_crawl_version

    LANGUAGE="amh"

    PATH_TO_INPUT_WARC = f"datasets/{cc_crawl_version}/{LANGUAGE}_warc_index"
    PATH_TO_FILES = natsorted(glob.glob(f"{PATH_TO_INPUT_WARC}/*.csv"))      # sorted

    PATH_TO_OUTPUT_FILES = f"datasets/{cc_crawl_version}/{LANGUAGE}_txt_collections_hf/" # output


    if not os.path.exists(PATH_TO_OUTPUT_FILES):
        os.makedirs(PATH_TO_OUTPUT_FILES)

    previously_done = check_last_folder(PATH_TO_OUTPUT_FILES)

    # end = 100
    end = (previously_done+2)*100
    count_folders = previously_done+1
    rows_to_skip = count_folders * 100


    only_lang = read_input(PATH_TO_FILES, LANGUAGE)
    only_lang_skipped = only_lang.iloc[rows_to_skip:].iterrows()

    # if all done already
    if rows_to_skip>len(only_lang):
        print('All already downloaded, exiting')
        exit()


    text_list = []

    # for index, row in tqdm(only_lang_skipped, total=len(only_lang)):
    #     # if index < (previously_done+1) * 100:
    #     #     continue
    #     text_from_html = download_and_extract(row)

    with multiprocessing.Pool(processes=24) as pool:
        rows_left = len(only_lang) - rows_to_skip
        for index, text_from_html in tqdm(pool.imap(download_and_extract, only_lang_skipped, chunksize=50), total=rows_left):
        
            if index < end:
                text_list.append(text_from_html)
            else:
                write_on_file(count_folders, text_list, PATH_TO_OUTPUT_FILES)
                text_list = []
                text_list.append(text_from_html)
                # This mean index = end.
                count_folders += 1
                end += 100             

    # to write the last batch -- 25246
    # Changes this to text_list instead of text_from_html, because text_from_html will only be the last webpage, 25246
    # But we need to write everything in the last batch, from 25200 to 25246 webpages
    if index < end:
        write_on_file(count_folders, text_list, PATH_TO_OUTPUT_FILES)


if __name__=='__main__':
    main()