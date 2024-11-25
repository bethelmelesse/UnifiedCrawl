#!/usr/bin/env bash

LANGUAGE="amh"
# CC_CRAWL_VERSION="CC-MAIN-2023-06"

grep "Error" datasets/$CC_CRAWL_VERSION/errors.txt | grep --only-matching "part-00[0-9]*" | grep --only-matching "[1-9][0-9]*" > datasets/$CC_CRAWL_VERSION/errors_todo.txt
mapfile -t error_counters < datasets/$CC_CRAWL_VERSION/errors_todo.txt


COUNTER=0

while IFS= read -r LINE; do          # print from file line by line
    if [[ " ${error_counters[*]} " =~ " ${COUNTER} " ]]; then
    echo $LINE
    $HOME/opt/duckdb -c "
        LOAD httpfs;
        LOAD parquet;

        SET s3_region='us-east-1';
        SET s3_access_key_id='$S3_ACCESS_KEY';
        SET s3_secret_access_key='$S3_SECRET_ACCESS_KEY';

        COPY (select url, content_languages, warc_filename, warc_record_offset, warc_record_length from PARQUET_SCAN('s3://commoncrawl/$LINE') where content_languages ilike '${LANGUAGE}%') TO 'datasets/$CC_CRAWL_VERSION/${LANGUAGE}_warc_index/$COUNTER.csv' (DELIMITER ',', HEADER TRUE);
        "
        sleep 1
    fi
    COUNTER=$((COUNTER+1))
done < datasets/$CC_CRAWL_VERSION/cc-index-table-warc.paths






