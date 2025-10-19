## MConAI
**MConAI: Multilingual Conversational AI for Low-Cost Early Alzheimer's Detection**

This project implements an Alzheimer's disease detection using model **BERT Multilingual** (mBERT) to classify conversational transcripts from multiple languages (English, Mandarin, Greek, Spanish). It supports stratified train-test splits and outputs evaluation metrics per dataset and language.

---

### Features

- Automatic transcript parsing (`.cha` files)
- Diagnosis labeling across multiple datasets (Delaware, Lu, Pitt, WLS, Chou, PerLA, Ivanova)
- Supports four classes: `AD`, `DM`, `HC`, `MCI`
- Multilingual classification using **BERT multilingual cased**
- Stratified train-test splitting for stable evaluation
- Computes standard metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC
- Handles datasets with small sample sizes gracefully
- Easily extendable to new datasets
---

### Dependencies

- Python 3.11+
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- tqdm
- openpyxl

---

### Quick Start

#### 1. Go to Your Project Folder

```bash
cd /Users/srihith/src/M2ConAI/src
```

#### 2. Create a Virtual Environment
```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv /Users/srihith/src/M2ConAI/src/venv
```

#### 3. Activate the Virtual Environment
```bash
source /Users/srihith/src/M2ConAI/src/venv/bin/activate
```

Check Python version:
```bash
python --version
```
#### 4. Install Required Packages
```bash
# Upgrade pip and install required packages
pip install --upgrade pip
pip install --upgrade pandas numpy scikit-learn torch tqdm transformers datasets accelerate openpyxl
```
#### 5. Run the Model
```bash
python model.py
```
#### 6. Deactivate Virtual Environment When Done
```bash
deactivate
```
### Sample Output
#### Label Summary

| Label | Count |
|-------|-------|
| HC    | 2151  |
| AD    | 1144  |
| MCI   | 339   |
| DM    | 79    |

**Label classes used for modeling:** `['AD', 'DM', 'HC', 'MCI']`


#### 80/20 Train-Test Split by Language

| Language  | Train_AD | Train_MCI | Train_HC | Test_AD | Test_MCI | Test_HC |
|-----------|----------|-----------|----------|---------|----------|---------|
| Mandarin  | 0        | 112       | 96       | 0       | 29       | 24      |
| English   | 834      | 89        | 1468     | 209     | 22       | 367     |
| Spanish   | 81       | 69        | 157      | 20      | 18       | 39      |


#### Final Evaluation Table

| Dataset   | Language  | Accuracy | Precision | Recall | F1-score |
|-----------|----------|----------|-----------|--------|----------|
| Chou      | Mandarin | 0.55     | 0.27      | 0.50   | 0.35     |
| Lu        | Mandarin | 1.00     | 1.00      | 1.00   | 1.00     |
| Delaware  | English  | 0.64     | 0.32      | 0.50   | 0.39     |
| Lu        | English  | 0.45     | 0.23      | 0.50   | 0.31     |
| WLS       | English  | 1.00     | 1.00      | 1.00   | 1.00     |
| Pitt      | English  | 0.88     | 0.80      | 0.86   | 0.82     |
| PerLA     | Spanish  | 1.00     | 1.00      | 1.00   | 1.00     |
| Ivanova   | Spanish  | 0.68     | 0.34      | 0.50   | 0.41     |
| **Average** |        | 0.74     | 0.57      | 0.69   | 0.61     |



#### How to Add New Datasets

Add .cha transcript files into a new language or dataset folder.

Update the get_diagnosis_label() function in model.py with rules for your new dataset.

Make sure the labels match one of the supported classes: AD, DM, HC, MCI.

Run python model.py to include the new dataset in training and evaluation.

### License

MIT License

### Author

Srihith Chennareddy
