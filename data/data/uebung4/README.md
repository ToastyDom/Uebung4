# NLP-gestützte Data Science
## Übung 4
### Datensatz
Der vorliegende Datensatz wurde aus der [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) erstellt.
Die TSV-Dateien enthalten das Sentiment Label aus dem Bereich [0, 4], wobei die Werte jeweils das Sentiment des Satzes wie folgt darstellen:

| Label | Bedeutung    |
|-------|--------------|
| 0     | sehr negativ |
| 1     | negativ      |
| 2     | neutral      |
| 3     | positiv      |
| 4     | sehr positiv |

Der Datensatz ist in drei Dateien aufgeteilt: `train.tsv`, `dev.tsv` und `test.tsv`.
Es sollte ausschließlich auf dem `train` Datensatz trainiert werden, der `dev` Datensatz sollte zum Optimieren von Hyperparametern verwendet werden und der `test` Datensatz ausschließlich zur Evaluation eines Modells nach Abschluss des Trainings.

#### Label-Verteilung in den Datensatz-Splits

| Label | Train | Dev | Test |
|-------|-------|-----|------|
| 0     | 1092  | 139 | 279  |
| 1     | 2218  | 289 | 633  |
| 2     | 1624  | 229 | 389  |
| 3     | 2322  | 279 | 510  |
| 4     | 1288  | 165 | 399  |

### Referenzen
```
Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
```