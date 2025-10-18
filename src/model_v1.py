import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ---------- 0. VAS METADATA LOADER ----------
def load_vas_metadata(parent_dir):
    vas_map = {}
    metadata_path = os.path.join(parent_dir, "english/vas/0demo.xlsx")
    if os.path.exists(metadata_path):
        df = pd.read_excel(metadata_path)
        for _, row in df.iterrows():
            fname = str(row.get("FileName", "")).lower()
            diag = str(row.get("Diagnosis", "")).upper()
            if "DEMENT" in diag or diag == "AD":
                label = "DM"
            elif "MCI" in diag:
                label = "MCI"
            elif "CONTROL" in diag or "HC" in diag:
                label = "HC"
            else:
                label = "Unknown"
            vas_map[fname.split(".")[0]] = label
        print(f"Loaded {len(vas_map)} entries from VAS metadata.")
    else:
        print("VAS metadata file (0demo.xlsx) not found.")
    return vas_map

# ---------- 1. AUTO LABEL RULES ----------
def get_diagnosis_label(filepath, vas_labels):
    path = filepath.replace("\\", "/").lower()
    fname = os.path.basename(path).lower().split(".")[0]

    if "english/delaware/control" in path: return "HC", "English", "Delaware"
    elif "english/delaware/mci" in path: return "MCI", "English", "Delaware"
    elif "english/lu/control" in path: return "HC", "English", "Lu"
    elif "english/lu/dementia" in path: return "DM", "English", "Lu"
    elif "english/pitt/control" in path: return "HC", "English", "Pitt"
    elif "english/pitt/dementia" in path: return "AD", "English", "Pitt"
    elif "english/wls" in path: return "HC", "English", "WLS"
    elif "english/vas" in path:
        for key in vas_labels.keys():
            if key in fname: return vas_labels[key], "English", "VAS"
        return "Unknown", "English", "VAS"

    elif "greek/dem@care" in path:
        if "/ad" in path: return "AD", "Greek", "Dem@Care"
        elif "/hc" in path: return "HC", "Greek", "Dem@Care"
        elif "/mci" in path: return "MCI", "Greek", "Dem@Care"
        else: return "Unknown", "Greek", "Dem@Care"

    elif "mandarin/chou/hc" in path: return "HC", "Mandarin", "Chou"
    elif "mandarin/chou/mci" in path: return "MCI", "Mandarin", "Chou"
    elif "mandarin/lu" in path: return "DM", "Mandarin", "Lu"

    elif "spanish/ivanova/hc" in path: return "HC", "Spanish", "Ivanova"
    elif "spanish/ivanova/mci" in path: return "MCI", "Spanish", "Ivanova"
    elif "spanish/perla" in path: return "AD", "Spanish", "PerLA"

    else: return "Unknown", "Unknown", "Unknown"

# ---------- 2. PARSER ----------
def parse_cha(file_path, label, language, dataset_name, metadata=None):
    texts = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith("*PAR:"):
                text = line[5:].strip()
                if text: texts.append(text)
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
                metadata = {"source_folder": os.path.basename(root)}
                data.append(parse_cha(file_path, label, language, dataset_name, metadata))
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
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
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
    try:
        auc = roc_auc_score(pd.get_dummies(labels), pd.get_dummies(preds))
    except:
        auc = 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

# ---------- 6. MAIN ----------
def main():
    parent_dir = "/Users/srihith/src/dementia"
    vas_labels = load_vas_metadata(parent_dir)
    all_data = load_all_cha(parent_dir, vas_labels)
    df = pd.DataFrame(all_data)

    allowed_labels = ['AD', 'DM', 'HC', 'MCI']
    df = df[df['label'].isin(allowed_labels)].reset_index(drop=True)

    # REMOVE German dataset completely
    df = df[df['language'] != 'German'].reset_index(drop=True)

    if df.empty:
        print("No transcripts found.")
        return

    # Label encoding
    label_encoder = LabelEncoder()
    df['label_enc'] = label_encoder.fit_transform(df['label'])
    num_labels = len(label_encoder.classes_)

    print("\n--- Label Summary ---")
    print(df['label'].value_counts())
    print("Label classes used for modeling:", list(label_encoder.classes_))

    # --- 80/20 Train-Test Split Table by Language ---
    split_distribution = []
    for lang in df['language'].unique():
        lang_subset = df[df['language']==lang]
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

    # --- Prepare tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=num_labels
    )

    eval_results = []

    # --- Train/test per dataset/language ---
    for lang in df['language'].unique():
        for ds in df[df['language']==lang]['dataset'].unique():
            subset = df[(df['language']==lang) & (df['dataset']==ds)]
            if len(subset) < 4:
                print(f"Skipping {ds} ({lang}) — too few samples")
                continue

            texts = subset['text'].tolist()
            labels = subset['label_enc'].tolist()

            # Stratified split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    texts, labels, test_size=0.2, stratify=labels, random_state=42
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    texts, labels, test_size=0.2, random_state=42
                )

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
                metric_for_best_model="accuracy"
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

            # Evaluate on unseen test set
            preds_output = trainer.predict(test_dataset)
            logits = preds_output.predictions
            preds = logits.argmax(axis=-1)
            metrics = compute_metrics((logits, y_test))

            eval_results.append({
                "Dataset": ds,
                "Language": lang,
                "Accuracy": round(metrics["accuracy"], 2),
                "Precision": round(metrics["precision"], 2),
                "Recall": round(metrics["recall"], 2),
                "F1-score": round(metrics["f1"], 2),
                "AUC-ROC": round(metrics["auc"], 2)
            })

    # Average metrics
    eval_df = pd.DataFrame(eval_results)
    avg_metrics = eval_df[["Accuracy","Precision","Recall","F1-score","AUC-ROC"]].mean()
    avg_row = {"Dataset":"Average","Language":""}
    avg_row.update({k: round(v,2) for k,v in avg_metrics.items()})
    eval_df = pd.concat([eval_df, pd.DataFrame([avg_row])], ignore_index=True)

    print("\n--- Final Evaluation Table ---")
    print(eval_df.to_string(index=False))


if __name__ == "__main__":
    main()


