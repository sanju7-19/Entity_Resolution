\# Amazon ML Entity Resolution



\## Overview



This project explores machine learning techniques for entity resolution: identifying records that refer to the same real-world entity across different data sources.



The workflow includes data inspection, validation, normalization, candidate-pair generation, feature engineering, model training, and evaluation.



\## Objectives



\* Inspect and validate input datasets.

\* Normalize fields to improve record comparability.

\* Develop and benchmark candidate-generation strategies.

\* Analyze candidate-generation failures and missed matching pairs.

\* Build training features for entity matching.

\* Train LightGBM models for matching decisions.



\## Technology Stack



\* Python

\* LightGBM

\* Pandas

\* NumPy

\* PyArrow

\* Parquet

\* SQLite



Verify these dependencies against your actual scripts before finalizing this list.



\## Project Structure



```text

amazon\_ml/

├── models/

│   ├── lgbm\_s2.txt

│   ├── lgbm\_s3.txt

│   ├── thresholds.json

│   └── training\_summary.csv

├── scripts/

│   ├── 01\_inspect\_data.py

│   ├── 01b\_profile\_fields.py

│   ├── 02\_inventory\_validate.py

│   ├── 03\_integrity\_check.py

│   ├── 04\_normalize\_raw.py

│   ├── 05\_normalization\_qa.py

│   ├── 06\_blocking\_benchmark.py

│   ├── 07\_blocking\_sample.py

│   ├── 08\_blocking\_failure\_analysis.py

│   ├── 09\_inspect\_missed\_pairs.py

│   ├── 10\_blocking\_v2\_benchmark.py

│   ├── 11\_combined\_blocking\_benchmark.py

│   ├── 12\_analyze\_combined\_misses.py

│   ├── 13\_blocking\_v3\_benchmark.py

│   ├── 14\_build\_training\_features.py

│   └── 15\_train\_lightgbm.py

├── .gitignore

├── README.md

└── requirements.txt

```



\## Methodology



\### 1. Data Inspection and Validation



Inspect input fields, profile records, and validate data integrity.



\### 2. Data Normalization



Normalize source records to improve consistency during record comparison.



\### 3. Candidate Generation



Benchmark blocking strategies to reduce the number of record pairs requiring detailed comparison.



\### 4. Failure Analysis



Analyze missed matching pairs and investigate limitations in candidate generation.



\### 5. Feature Engineering



Generate features that help distinguish matching records from non-matching records.



\### 6. Model Training



Train LightGBM models for entity-matching classification.



\## Dataset



The dataset and generated intermediate files are excluded from this repository to avoid publishing large files.



Refer to the original dataset source and its usage restrictions before downloading or redistributing any data.



\## Model Artifacts



The `models/` directory contains trained model artifacts and associated configuration or training-summary files.



\## Reproducibility



The project scripts and dependency information should be used to document the workflow.



Exact installation and execution commands will be added after validating the scripts and their dependencies.



\## Evaluation



Add the actual evaluation results from your experiments, including relevant precision, recall, F1-score, and candidate-generation recall where available.



Do not report unmeasured results.



\## Future Improvements



\* Improve candidate-generation recall.

\* Evaluate feature-engineering strategies.

\* Tune classification thresholds.

\* Compare model performance using held-out data.

\* Document reproducible evaluation procedures.



