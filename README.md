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

