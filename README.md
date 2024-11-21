# Amharic Data Collection, Bechmarking, and Fine-tuning

## How to setup an environment

### Install conda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Create conda python environment, install cuda

```bash
conda create -y --name data_env python==3.10.12
conda activate data_env
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
# install gcc compiler
sudo apt update && sudo apt install -y gcc unzip
```

### Install python dependencies

```bash
pip install -r setup/requirements_data.txt -r setup/requirements.txt
# Alternatively, to exactly mirror our dependency environment, use below -
# pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r setup/requirments_full.txt

```

<!-- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tqdm pandas evaluate bitsandbytes accelerate openpyxl sentencepiece scipy peft tensorboard
pip install -r setup/requirments_full.txt -->

### Setup other dependencies to download CC data

```bash
# install duckdb
mkdir ~/opt
cd ~/opt
wget https://github.com/duckdb/duckdb/releases/download/v0.8.1/duckdb_cli-linux-amd64.zip
unzip ./duckdb_cli-linux-amd64.zip
~/opt/duckdb -c "INSTALL httpfs"
```

## How to download the data from CC

### Filter the CC index by Language

Run the following in the terminal-
Set the environment variable `CC_CRAWL_VERSION` before running!

```bash
CC_CRAWL_VERSION="CC-MAIN-2023-06" ./download_data/download_and_filter_warc_index.sh 2>&1 | tee datasets/errors.txt
```

The above command sometimes has some errors, so just re-run the following script many many times -

```bash
CC_CRAWL_VERSION="CC-MAIN-2023-06" ./download_data/download_and_filter_warc_index_retry.sh 2>&1 | tee datasets/errors.txt
CC_CRAWL_VERSION="CC-MAIN-2023-06" ./download_data/download_and_filter_warc_index_retry.sh 2>&1 | tee datasets/errors.txt
```

## Remove duplicate URLs

If your language has many URLs repeatedly crawled, you may benefit for removing duplicate URLs in the warc files across all dumps.

This can be done via simple use of the `pandas` library for example. We skip this code here

### Download and extract the text from filtered CC index

```bash
python ./download_data/download_and_extract_text_from_warc.py --cc-crawl-version=CC-MAIN-2023-06
```

## Deduplicate Data

### Install Rust

```bash
curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env

# check if rust is installed successfully
rustc --version
```

### Build binary for deduplicate-texts-datasets

Note - Increase/Decrease the variable `jobs_at_once` in the file [make_suffix_array.py](deduplicate_data/deduplicate-text-datasets/scripts/make_suffix_array.py) to increase/decrease the number of parallel jobs based on your CPU cores. Decreasing the number of parallel jobs may also help reducing RAM usage.

```bash
cd ./deduplicate_data/deduplicate-text-datasets
cargo build
```

### Run Data Deduplication

#### Combine crawl files into one for each crawl

First, combine the files in a single crawl into one file. The following command will do it for all crawls.

```bash
python deduplicate_data/combine_single_dump.py
```

#### Deduplicate a single crawl

```bash
cd deduplicate_data/text-dedup

# remove any previous deduplicated files
rm -rf ./output && rm -rf ../deduplicate-text-datasets/.cache/ ../deduplicate-text-datasets/output/ ../deduplicate-text-datasets/tmp

# "(When running on larger files, if you get an error that you have too many open files, that's because this script opens lots of files. You should run ulimit -Sn 1000000 to "fix" the error. You might want to do this preemptively before hitting this crash after hour ten of the job.)" 
ulimit -Sn 1000000

# To de-duplicate a single-crawl
CC_CRAWL="CC-MAIN-2023-06"
LANGUAGE="amh"
python -m text_dedup.suffix_array \
    --path "json" \
    --data_files "../../datasets/$CC_CRAWL/amh_txt_collections_hf_combined.jsonl" \
    --output "../../datasets/$CC_CRAWL/${LANGUAGE}_dedup" \
    --split 'train' \
    --column 'text' \
    --google_repo_path "../deduplicate-text-datasets" \
    --local \
    --batch_size 10000 \
    --k 50
```

We can use a simple bash for loop to run the above for all the crawls using -

```bash
LANGUAGE="amh"
for i in ../../datasets/CC-MAIN* ; do
    echo $i
    CC_CRAWL=`basename $i` 
    echo $CC_CRAWL
    rm -rf ./output && rm -rf ../deduplicate-text-datasets/.cache/ ../deduplicate-text-datasets/output/ ../deduplicate-text-datasets/tmp
    python -m text_dedup.suffix_array \
        --path "json" \
        --data_files "../../datasets/$CC_CRAWL/${LANGUAGE}_txt_collections_hf_combined.jsonl" \
        --output "../../datasets/$CC_CRAWL/amh_dedup" \
        --split 'train'\
        --column 'text' \
        --google_repo_path "../deduplicate-text-datasets" \
        --local \
        --batch_size 10000 \
        --k 50 ; 
    done
```

#### Deduplicate all crawls together

We deduplicated all the crawls separately first, because the original code requires much more time/memory if there are a very large number of duplicates. (see issue discussion [here](https://github.com/google-research/deduplicate-text-datasets/issues/18)).

We can run the command below to deduplicate across crawls our already-deduplicated single crawls - 

```bash
rm -rf ./output && rm -rf ../deduplicate-text-datasets/.cache/ ../deduplicate-text-datasets/output/ ../deduplicate-text-datasets/tmp
python -m text_dedup.suffix_array \
    --path "arrow" \
    --data_files "../../datasets/*/${LANGUAGE}_dedup/*.arrow" \
    --output "../../datasets/${LANGUAGE}_dedup" \
    --split 'train' \
    --column 'text' \
    --google_repo_path "../deduplicate-text-datasets" \
    --local \
    --batch_size 10000 \
    --k 50
```

#### Filter very short documents

After deduplication some documents become very short in length, hence, we remove those documents of length less than 100 characters. Change the `LANGUAGE` variable at the top of the file.

```bash
python deduplicate_data/remove_short_docs.py
```

## How to benchmark original models on amharic/english

1. Change the values in the file [evaluate_model/hyperparameters.py](evaluate_model/hyperparameters.py)
1. Run the model using `python evaluate_model/run_model.py`

## How to finetune model on amharic

Run the script below to finetune. Change the necessary variables.

```bash
# rm -rf ./finetune_model/output_dir/facebook/xglm-4.5B/
./finetune_model/run_finetune.sh
```
