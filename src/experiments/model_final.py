import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ---------- 0. Load VAS Metadata ----------
def load_vas_metadata(metadata_path, nrows=102):
    """
    Load VAS metadata, normalize IDs (ignore leading zeros),
    and return a dictionary mapping VAS ID -> label.
    """
    metadata_path = "/Users/srihith/src/dementia/english/vas/0demo.xlsx"

    # Read Excel
    df = pd.read_excel(metadata_path, usecols=["VAS ID", "H/MCI/D"], nrows=nrows)
    
    # Normalize VAS IDs: remove leading zeros
    df['VAS ID'] = df['VAS ID'].astype(str).str.lstrip('0')
    
    # Map label: H -> HC, D -> DM, MCI -> MCI
    label_map = {'H': 'HC', 'D': 'DM', 'MCI': 'MCI'}
    labels = []
    for idx, row in df.iterrows():
        if row['H/MCI/D'] == 'H':
            labels.append('HC')
        elif row['H/MCI/D'] == 'D':
            labels.append('DM')
        elif row['H/MCI/D'] == 'MCI':
            labels.append('MCI')
        else:
            labels.append('Unknown')
    
    df['Label'] = labels
    
    # Build dictionary: VAS ID -> label
    vas_dict = dict(zip(df['VAS ID'], df['Label']))
    return vas_dict


# ---------- 1. AUTO LABEL RULES ----------
def get_diagnosis_label(filepath, vas_labels):
    path = filepath.replace("\\", "/").lower()
    fname = os.path.basename(path).lower().split(".")[0].strip()

    # English
    if "english/delaware/control" in path: return "HC", "English", "Delaware"
    elif "english/delaware/mci" in path: return "MCI", "English", "Delaware"
    elif "english/lu/control" in path: return "HC", "English", "Lu"
    elif "english/lu/dementia" in path: return "DM", "English", "Lu"
    elif "english/pitt/control" in path: return "HC", "English", "Pitt"
    elif "english/pitt/dementia" in path: return "AD", "English", "Pitt"
    elif "english/wls" in path: return "HC", "English", "WLS"
    elif "english/vas" in path.lower():
        # extract file name without extension and remove leading zeros
        fname = os.path.splitext(os.path.basename(filepath))[0].lstrip('0')
        
        # check against normalized vas_labels keys
        if fname in vas_labels:
            return vas_labels[fname], "English", "VAS"
        else:
            print(f"[DEBUG] Unknown VAS file (not in metadata): {filepath}")
            return "Unknown", "English", "VAS"

    # Greek
    elif "greek/dem@care" in path or "greek/demcare" in path:
        if "/ad" in path or "ad" in path: return "AD", "Greek", "Dem@Care"
        elif "/hc" in path or "hc" in path: return "HC", "Greek", "Dem@Care"
        elif "/mci" in path or "mci" in path: return "MCI", "Greek", "Dem@Care"
        else: return "Unknown", "Greek", "Dem@Care"

    # Mandarin
    elif "mandarin/chou/hc" in path: return "HC", "Mandarin", "Chou"
    elif "mandarin/chou/mci" in path: return "MCI", "Mandarin", "Chou"
    elif "mandarin/lu" in path: return "DM", "Mandarin", "Lu"

    # Spanish
    elif "spanish/ivanova/hc" in path: return "HC", "Spanish", "Ivanova"
    elif "spanish/ivanova/mci" in path: return "MCI", "Spanish", "Ivanova"
    elif "spanish/ivanova/ad" in path: return "AD", "Spanish", "Ivanova"
    elif "spanish/perla" in path: return "AD", "Spanish", "PerLA"

    else:
        return "Unknown", "Unknown", "Unknown"


# ---------- 2. PARSER ----------
def parse_cha(file_path, label, language, dataset_name, metadata=None):
    texts = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("*PAR:"):
                text = line[5:].strip()
                if text: texts.append(text)
    if not texts:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            fallback_text = " ".join([line.strip() for line in f if line.strip()])
            if fallback_text:
                texts.append(fallback_text)

    data = {"text": " ".join(texts), "label": label, "language": language, "dataset": dataset_name}
    if metadata: data.update(metadata)
    return data


# ---------- 3. LOADER ----------
def load_all_cha(parent_dir, vas_labels):
    data = []
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if file.endswith(".cha"):
                file_path = os.path.join(root, file)
                label, language, dataset_name = get_diagnosis_label(file_path, vas_labels)
                if label == "Unknown":
                    print("[DEBUG] Unknown label:", file_path)
                metadata = {"source_folder": os.path.basename(root)}
                parsed = parse_cha(file_path, label, language, dataset_name, metadata)
                if parsed['text'].strip():
                    data.append(parsed)
    print(f"[INFO] Total transcripts loaded: {len(data)}")
    return data


# ---------- 4. DATASET ----------
class AlzheimerDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


# ---------- 5. METRICS ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------- 6. MAIN ----------
def main():
    parent_dir = "/Users/srihith/src/dementia"
    vas_labels = load_vas_metadata(parent_dir)
    all_data = load_all_cha(parent_dir, vas_labels)
    df = pd.DataFrame(all_data)

    allowed_labels = ['AD', 'DM', 'HC', 'MCI']
    df = df[df['label'].isin(allowed_labels)].reset_index(drop=True)

    if df.empty:
        print("No transcripts found.")
        return

    label_encoder = LabelEncoder()
    df['label_enc'] = label_encoder.fit_transform(df['label'])
    num_labels = len(label_encoder.classes_)

    print("\n--- Label Summary ---")
    print(df['label'].value_counts())
    print("Label classes used for modeling:", list(label_encoder.classes_))



    # ---------- 7. SIMPLE DATASET OVERVIEW ----------
    overview = (
        df.groupby(["language", "dataset", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Add placeholder columns for tasks (PD, FT, SR, FC, NA)
    overview["PD"] = "X"
    overview["FT"] = "X"
    overview["SR"] = "X"
    overview["FC"] = "X"
    overview["NA"] = "X"

    # Reorder columns to match desired layout
    overview = overview[
        ["language", "dataset", "PD", "FT", "SR", "FC", "NA", "DM", "AD", "MCI", "HC"]
    ]

    print("\n=== Dataset Overview ===")
    print(overview.to_string(index=False))

    # ---------- 8. 80/20 Train-Test Split ----------
    split_distribution = []
    for lang in df['language'].unique():
        lang_subset = df[df['language'] == lang]
        X_lang = lang_subset['text'].tolist()
        y_lang = lang_subset['label_enc'].tolist()
        try:
            X_train_lang, X_test_lang, y_train_lang, y_test_lang = train_test_split(
                X_lang, y_lang, test_size=0.2, stratify=y_lang, random_state=42
            )
        except ValueError:
            X_train_lang, X_test_lang, y_train_lang, y_test_lang = train_test_split(
                X_lang, y_lang, test_size=0.2, random_state=42
            )
        counts_train = pd.Series(y_train_lang).value_counts()
        counts_test = pd.Series(y_test_lang).value_counts()
        split_distribution.append({
            "Language": lang,
            "Train_AD": counts_train.get(label_encoder.transform(['AD'])[0], 0),
            "Train_MCI": counts_train.get(label_encoder.transform(['MCI'])[0], 0),
            "Train_HC": counts_train.get(label_encoder.transform(['HC'])[0], 0),
            "Test_AD": counts_test.get(label_encoder.transform(['AD'])[0], 0),
            "Test_MCI": counts_test.get(label_encoder.transform(['MCI'])[0], 0),
            "Test_HC": counts_test.get(label_encoder.transform(['HC'])[0], 0),
        })

    print("\n--- 80/20 Train-Test Split by Language ---")
    print(pd.DataFrame(split_distribution).to_string(index=False))

    # ---------- 9. Train/Test & Evaluation ----------
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=num_labels
    )

    eval_results = []
    k_folds = 5

    for lang in df['language'].unique():
        for ds in df[df['language'] == lang]['dataset'].unique():
            subset = df[(df['language'] == lang) & (df['dataset'] == ds)]
            if len(subset) < 4:
                print(f"Skipping {ds} ({lang}) — too few samples")
                continue

            texts = subset['text'].tolist()
            labels = subset['label_enc'].tolist()
            skf = StratifiedKFold(n_splits=min(k_folds, len(labels)), shuffle=True, random_state=42)
            fold_metrics = []

            for train_idx, test_idx in skf.split(texts, labels):
                X_train = [texts[i] for i in train_idx]
                X_test = [texts[i] for i in test_idx]
                y_train = [labels[i] for i in train_idx]
                y_test = [labels[i] for i in test_idx]

                train_dataset = AlzheimerDataset(X_train, y_train, tokenizer)
                test_dataset = AlzheimerDataset(X_test, y_test, tokenizer)

                training_args = TrainingArguments(
                    output_dir="./results",
                    num_train_epochs=3,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    logging_dir='./logs',
                    logging_steps=10,
                    learning_rate=2e-5,
                    save_total_limit=1,
                    load_best_model_at_end=True,
                    metric_for_best_model="accuracy",
                    disable_tqdm=True
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics
                )

                trainer.train()
                preds_output = trainer.predict(test_dataset)
                metrics = compute_metrics((preds_output.predictions, y_test))
                fold_metrics.append(metrics)

            avg_fold_metrics = {k: round(sum(d[k] for d in fold_metrics)/len(fold_metrics), 2) 
                                for k in fold_metrics[0].keys()}

            eval_results.append({
                "Dataset": ds,
                "Language": lang,
                **avg_fold_metrics
            })

    # ---------- 10. Evaluation Tables ----------
    eval_df = pd.DataFrame(eval_results).round(2)

    print("\n=== Dataset-Level Evaluation Table ===")
    print(eval_df.to_string(index=False))

    lang_table = (
        eval_df.groupby("Language")[["accuracy", "precision", "recall", "f1"]]
        .mean().reset_index().round(2)
    )
    avg_row = {"Language": "Average"}
    avg_row.update({col: round(lang_table[col].mean(), 2) for col in ["accuracy", "precision", "recall", "f1"]})
    lang_table = pd.concat([lang_table, pd.DataFrame([avg_row])], ignore_index=True)

    print("\n=== Language-Level Aggregated Evaluation Table ===")
    print(lang_table.to_string(index=False))

    eval_df.to_csv("results_dataset_level.csv", index=False)
    lang_table.to_csv("results_language_level.csv", index=False)
    eval_df.to_latex("results_dataset_level.tex", index=False, caption="Dataset-Level Evaluation Results", label="tab:dataset_results")
    lang_table.to_latex("results_language_level.tex", index=False, caption="Language-Level Aggregated Evaluation Results", label="tab:language_results")


if __name__ == "__main__":
    main()

