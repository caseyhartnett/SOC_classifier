# SOC_classifier
ML approach to classify job title for 2018 Standard Occupational Classification (SOC) system. 

###DVC Steps

Download Data from O*Net Center

```
dvc  run \
    -n download_onet_data \
    -o data/onet_data.txt \
    wget https://www.onetcenter.org/dl_files/database/db_20_1_text/Sample%20of%20Reported%20Titles.txt -O data/onet_data.txt
```


```
dvc  run \
    -n download_onet_job_titles \
    -o data/onet_job_titles.txt \
    wget https://www.onetcenter.org/dl_files/database/db_25_0_text/Occupation%20Data.txt -O data/onet_job_titles.txt
```

Modify Downloaded Data

```
dvc run \
    -n modify_data \
    -o data/modified_data.csv \
    python src/modify_data.py
```
