# GitHub Issue Classifier – Progetto Data Analytics

Progetto curriculare per il corso di **Data Analytics / Web Mining**.  
L'obiettivo è classificare automaticamente le GitHub Issues in due categorie — **bug** e **feature** — esplorando e confrontando cinque approcci progressivamente più sofisticati, dall'estrazione di features classica fino al fine-tuning di un modello transformer.

---

## Indice

1. [Descrizione del problema](#1-descrizione-del-problema)
2. [Dataset](#2-dataset)
3. [Struttura del repository](#3-struttura-del-repository)
4. [Pipeline e approcci](#4-pipeline-e-approcci)
5. [Risultati](#5-risultati)
6. [Requisiti e installazione](#6-requisiti-e-installazione)
7. [Come eseguire i notebook](#7-come-eseguire-i-notebook)

---

## 1. Descrizione del problema

Le piattaforme di sviluppo software come GitHub ricevono migliaia di issues al giorno. Classificarle manualmente in *bug* (comportamento inatteso, crash, errore) o *feature* (richiesta di nuova funzionalità, miglioramento) è dispendioso. L'obiettivo del progetto è costruire e confrontare diversi classificatori automatici basati sul testo del titolo e del body dell'issue.

---

## 2. Dataset

Il dataset utilizzato è un insieme di GitHub Issues raccolte da repository pubblici, salvato in formato `.parquet` per efficienza.

| Proprietà | Valore |
|---|---|
| Formato sorgente | `github_issues_dataset.parquet` |
| Feature utilizzate | `title`, `body` |
| Target | `label` → `bug` / `feature` |
| Classi originali | Molteplici label raw (es. `bug`, `enhancement`, `help wanted`, …) |
| Mapping applicato | Tutte le label mappate a 3 macro-classi; righe non classificabili rimosse |
| Classi finali | **bug**, **feature** (binario) |

Il dataset è stato diviso in train e test set e salvato in `data/train_set.csv` e `data/test_set.csv`.

> **Nota:** i file `.parquet` e `.npy` di grandi dimensioni non sono inclusi nel repository (vedere `.gitignore`). Rigenera i file intermedi eseguendo i notebook in ordine.

---

## 3. Struttura del repository

```
ProgettoData_Web/
│
├── data/
│   ├── github_issues_dataset.parquet   # dataset originale
│   ├── github_issues_clean.csv         # versione pulita dopo EDA
│   ├── train_set.csv                   # training set
│   ├── test_set.csv                    # test set
│   ├── X_train_distilbert.npy          # embeddings DistilBERT (train)
│   └── X_test_distilbert.npy           # embeddings DistilBERT (test)
│
├── notebooks/
│   ├── 01_eda.ipynb                        # Analisi esplorativa dei dati
│   ├── 02_TF-IDF.ipynb                     # Baseline con TF-IDF + classificatori classici
│   ├── 03_WordEmbeddings.ipynb             # Word2Vec custom e GloVe pretrained
│   ├── 04_DistilBertEmbeddings.ipynb       # Embeddings DistilBERT + classificatori
│   ├── 05_Classificatore_Fine_Tuned.ipynb  # Valutazione modello fine-tuned
│   ├── 06_Conclusioni.ipynb                # Confronto globale e visualizzazioni finali
│   └── Fine_Tuning_Modello.ipynb           # Script di fine-tuning (training loop)
│
├── models/
│   └── github_model_FINETUNED/         # Pesi e tokenizer del modello fine-tuned
│       ├── config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── results/
│   ├── tfidf_results.json
│   ├── word2vec_results.json
│   ├── glove_results.json
│   ├── metrics_distilbert_embeddings.json
│   └── metrics_bert.json
│
├── figures/                            # Tutti i grafici generati dai notebook
│
└── README.md
```

---

## 4. Pipeline e approcci

I notebook seguono una progressione logica dalla rappresentazione testuale più semplice alla più complessa:

### Notebook 01 – EDA
Analisi esplorativa: distribuzione delle classi, lunghezza dei testi, valori mancanti, mapping delle label raw nelle macro-classi `bug` e `feature`. Pulizia del dataset e creazione di train/test set.

### Notebook 02 – TF-IDF
Approccio baseline classico. Il testo viene preprocessato (lowercase, rimozione punteggiatura, stopwords, stemming) e vettorizzato con **TF-IDF**. Vengono testati tre classificatori:
- Logistic Regression
- Decision Tree
- SVM

### Notebook 03 – Word Embeddings
Due varianti di rappresentazione densa del testo:
- **Word2Vec custom**: addestrato sul corpus del dataset
- **GloVe pretrained**: vettori pre-addestrati su Wikipedia + Gigaword

Per entrambi, il vettore di un documento è la media dei vettori delle parole. Stessi tre classificatori del notebook 02.

### Notebook 04 – DistilBERT Embeddings
Gli embedding contestuali di **DistilBERT** (modello transformer pretrained) vengono estratti e usati come feature fisse, senza modificare i pesi del modello. Classificatori: Logistic Regression, Decision Tree, SVM.

### Notebook 05 – Fine-tuning
Il modello **DistilBERT** viene fine-tuned end-to-end sul task di classificazione binaria usando la libreria HuggingFace Transformers. Il modello fine-tuned è salvato in `models/github_model_FINETUNED/` e valutato sul test set.

### Notebook 06 – Conclusioni
Caricamento di tutti i risultati JSON, tabella comparativa ordinata per F1-score, e visualizzazioni: grouped bar chart, heatmap delle metriche, progression line chart, box plot, confusion matrices.

---

## 5. Risultati

Riepilogo delle performance migliori per ciascun approccio (F1-score macro sul test set completo):

| Approccio | Miglior classificatore | Accuracy | F1-score |
|---|---|---|---|
| TF-IDF | SVM | 0.886 | 0.886 |
| Word2Vec custom | SVM | 0.877 | 0.877 |
| GloVe pretrained | SVM | 0.831 | 0.831 |
| DistilBERT Embeddings | SVM | 0.881 | 0.881 |
| **BERT Fine-tuned** | – | **0.900** | **0.901** |

Il modello fine-tuned raggiunge le performance migliori, confermando che adattare i pesi di un transformer pretrained al task specifico supera tutti gli approcci basati su feature fisse. Tra gli approcci classici, TF-IDF con SVM si dimostra sorprendentemente competitivo rispetto agli embedding più complessi.

---

## 6. Requisiti e installazione

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
pip install torch transformers datasets
pip install gensim                   # Word2Vec
pip install fastparquet              # lettura .parquet
pip install jupyter
```

---

## 7. Come eseguire i notebook

I notebook vanno eseguiti **in ordine**, perché ciascuno produce file (CSV, NPY, JSON) usati dai successivi.

```
01 → 02 → 03 → 04 → Fine_Tuning_Modello → 05 → 06
```

> `Fine_Tuning_Modello.ipynb` va eseguito prima del 05 perché produce i pesi del modello in `models/github_model_FINETUNED/`.

Per avviare Jupyter:

```bash
cd notebooks/
jupyter notebook
```

### Tempi di esecuzione stimati (Mac M2)

| Notebook | Tempo stimato |
|---|---|
| 01 EDA | ~2 min |
| 02 TF-IDF | ~5 min |
| 03 Word Embeddings | ~10 min |
| 04 DistilBERT Embeddings | ~15 min |
| Fine_Tuning_Modello | ~30-60 min |
| 05 Classificatore fine-tuned | ~5 min |
| 06 Conclusioni | ~2 min |