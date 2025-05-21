import json
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Config
device = "cuda" if torch.cuda.is_available() else "cpu"
max_samples = 500  # nombre de textes de chaque catégorie
checkpoint_path = "detector-base.pt"
fake_file = "data/small-117M-k40.test.jsonl"
real_file = "data/webtext.test.jsonl"

# Chargement modèle et tokenizer
print("Chargement du modèle...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval().to(device)

# Fonction de prédiction
def predict_label(text):
    tokens = tokenizer.encode(text, max_length=tokenizer.model_max_length - 2, truncation=True)
    input_ids = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
    return prediction  # 0 = IA, 1 = humain

# Chargement des textes
def load_texts(path, label, max_samples):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                data = json.loads(line)
                texts.append(data["text"])
                labels.append(label)
            except Exception as e:
                print("Erreur lors du chargement :", e)
    return texts, labels

# Lecture des fichiers
print("Chargement des textes IA et humains...")
fake_texts, fake_labels = load_texts(fake_file, 0, max_samples)
real_texts, real_labels = load_texts(real_file, 1, max_samples)
texts = fake_texts + real_texts
true_labels = fake_labels + real_labels

# Prédiction
print("Début des prédictions...")
predicted_labels = []
for text in tqdm(texts):
    try:
        predicted_labels.append(predict_label(text))
    except Exception as e:
        print("Erreur de prédiction :", e)
        predicted_labels.append(0)

# Évaluation
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Résultat
print("\n=== ÉVALUATION DU MODÈLE VERITEXT ===")
print(f"Textes évalués : {len(texts)}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Précision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
