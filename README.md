# Common Differences in the Notebook Depending on the Number of Classes
## Dataset
The dataset is available for download at the link:
[Download Dataset.csv](https://drive.google.com/file/d/14VVRGvPeKVbza4GL66l41JZCpTN2Pq7G/view?usp=sharing)

## 1. NLP Augmentation

- In the notebooks without the NLP augmentation approach, the NLP augmentation part should be commented out.
- For the three-class approach and binary classification, `augment_minority_class` is used.
- For Multi-Class Classification (five classes), `augment_all_minority_classes` is used.

## 2. Model-Based Methods for Addressing Class Imbalance; Model Training

### Multi-Class Approaches (3 and 5 classes):

```python
'Random Forest': 
    'model': RandomForestClassifier(random_state=42, class_weight='balanced')

'XGBoost': 
    'model': XGBClassifier(eval_metric="mlogloss", random_state=42, 
                           objective='multi:softmax', num_class=3)  # Adjust to 3 or 5

'Logistic Regression': 
    'model': LogisticRegression(random_state=42, class_weight='balanced', 
                                multi_class='multinomial', max_iter=1000)
```

### Binary Classification:

```python
'Random Forest': 
    'model': RandomForestClassifier(random_state=42, class_weight='balanced')

'XGBoost':
    'model': XGBClassifier(eval_metric="logloss", random_state=42, 
                           scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))

'Logistic Regression': 
    'model': LogisticRegression(random_state=42, multi_class='multinomial', max_iter=1000)
```

## 3. Resampling Techniques; Model Training

### Multi-Class Classification: 3 Classes

```python
resamplers = {
    'SMOTE': SMOTE(sampling_strategy={0: 5000, 1: 5800, 2: 7072}, k_neighbors=5, random_state=42),
    
    'SMOTETomek': SMOTETomek(smote=SMOTE(sampling_strategy={0: 5000, 1: 5800, 2: 7072}, 
                                          k_neighbors=5, random_state=42), random_state=42),
    
    'Tomek Links': TomekLinks(sampling_strategy='auto'),  # No need for sampling_strategy as it's handled automatically
}
```

### Multi-Class Classification: 5 Classes

```python
resamplers = {
    'SMOTE': SMOTE(
        sampling_strategy={0: 4000, 1: 5000, 2: 22612, 3: 5000, 4: 6965},  
        k_neighbors=3,  
        random_state=42
    ),
    
    'SMOTETomek': SMOTETomek(
        smote=SMOTE(
            sampling_strategy={0: 4000, 1: 5000, 2: 22612, 3: 5000, 4: 6965}, 
            k_neighbors=3,  
            random_state=42
        ),
        random_state=42
    ),
    
    'Tomek Links': TomekLinks(sampling_strategy='not minority')
}
```

### Binary Classification: 2 Classes

```python
resamplers = {
    'SMOTE': SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42),
    'SMOTETomek': SMOTETomek(smote=SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42), random_state=42),
    'Tomek Links': TomekLinks(sampling_strategy='auto'),
}
```

### Model Training for Multi-Class (3 and 5 classes):

```python
'Random Forest': 
    'model': RandomForestClassifier(random_state=42)

'XGBoost': 
    'model': XGBClassifier(random_state=42, objective='multi:softmax', num_class=3)  # Adjust to 3 or 5
```

### Model Training for Binary Classification:

```python
'Random Forest': 
    'model': RandomForestClassifier(random_state=42)

'XGBoost': 
    'model': XGBClassifier(random_state=42, objective='binary:logistic')

'Logistic Regression': 
    'model': LogisticRegression(random_state=42, multi_class='multinomial', max_iter=1000)
```
