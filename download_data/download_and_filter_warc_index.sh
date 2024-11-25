#!/usr/bin/env bash

LANGUAGE="amh"
# CC_CRAWL_VERSION="CC-MAIN-2023-06"

PATH_INDEX="https://data.commoncrawl.org/crawl-data/$CC_CRAWL_VERSION/cc-index-table.paths.gz"  # Columnar URL index files - tells you which website is where in the crawl

# Make folder for filtered warc
mkdir -p datasets/$CC_CRAWL_VERSION/${LANGUAGE}_warc_index/

curl -o datasets/$CC_CRAWL_VERSION/cc-index-table.paths.gz "$PATH_INDEX"   # download file using path
gzip -d datasets/$CC_CRAWL_VERSION/cc-index-table.paths.gz -f           # decompress the file
grep --fixed-strings "subset=warc" datasets/$CC_CRAWL_VERSION/cc-index-table.paths > datasets/$CC_CRAWL_VERSION/cc-index-table-warc.paths       # filter warc files

COUNTER=0

while IFS= read -r LINE; do          # print from file line by line
   echo "$LINE"
   "$HOME"/opt/duckdb -c "
    LOAD httpfs;
    LOAD parquet;

    SET s3_region='us-east-1';
    SET s3_access_key_id='$S3_ACCESS_KEY';
    SET s3_secret_access_key='$S3_SECRET_ACCESS_KEY';

    COPY (select url, content_languages, warc_filename, warc_record_offset, warc_record_length from PARQUET_SCAN('s3://commoncrawl/$LINE') where content_languages ilike '${LANGUAGE}%') TO 'datasets/$CC_CRAWL_VERSION/${LANGUAGE}_warc_index/$COUNTER.csv' (DELIMITER ',', HEADER TRUE);
    "                                    # get the url, ... and save it to a file -- "common crawl splits urls,.. file by file"
    COUNTER=$((COUNTER+1))
    # sleep 3
done < datasets/$CC_CRAWL_VERSION/cc-index-table-warc.paths         # read this file
