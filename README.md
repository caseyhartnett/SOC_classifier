# SOC_classifier

ML approach to classify job title for 2018 Standard Occupational Classification (SOC) system.

### Approach

The project will utilize SkLearn's [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to transform the job title into an array based on the words contained.

This is then followed with the [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) to classify based on the vectorized job title.

SkLearn's [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) is used to wrap the models TfidfVectorizer and RandomForestClassifier together. Allowing for easy use and storage of the model in combination and not separate with custom code. 

Additionally, [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) is utalized to perform a basic hyper parameter search and cross validation during model building. This is parallelized allowing for automatic scaling based on machine training models. 

### Results

Model does not work well for individual occupations. This is reasonable since there are nearly thousand occupations in the dataset. 

Model does have some quality performance for `Major Groups` and `High Level Groups` (22 class and 12 classes respectively). The notebook `PipelineExamples.ipynb` show the accuracy and confusion matrix for the resulting models. 

### Future work

* More training data that is hand labeled to increase accuracy
* Removal of groups and occupations like for the Military. 
* Separate model for acronym usage like `CEO`, `CFO`, `FNP`, `RN` etc.
* API and deployment for public use.

### Install and Run Steps

From within the project run the following. This will setup the pipenv environment for the project based on the requirements. 

```pipenv install --dev```

Then to reproduce work run the following. This will then run the DVC steps to reproduce the models.

```dvc repro classify_and_pickle```

### DVC Steps

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
    -d data/onet_job_titles.txt -d data/onet_data.txt \
    -d src/modify_data.py \
    -o data/modified_data.csv \
    python src/modify_data.py
```

Pickle Models

```
dvc run \
    -n classify_and_pickle \
    -d data/modified_data.csv -d src/classify_and_pickle.py  -d src/soc_classifier.py \
    -o models/soc_code_model.pickle -o models/soc_high_level_model.pickle -o models/soc_major_model.pickle \
    python src/classify_and_pickle.py
```
