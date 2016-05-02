# Vectorizer and optimization


# Vectorizer 

``` python 
 class Vectorizer(conf=None, featurizer=None, normalizers=None, decomposers=None) 
```

Small text mining vectorizer.

The purpose of this class is to provide a small set of methods that can
operate on the data provided in the Experiment class. It can load data
from an iterator or .csv, and guides that data along a set of modules such
as the feature extraction, tf*idf function, SVD, etc. It can be controlled
through a settings dict that is provided in conf.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | conf | dict | Configuration dictionary passed to the experiment class. |
        

| Attributes    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | conf | dict |         Configuration dictionary passed to the experiment class. |
        | hasher | class |         Configuration dictionary passed to the experiment class.        DictVectorizer class from sklearn. |
        | decomposers | class | Configuration dictionary passed to the experiment class.DictVectorizer class from sklearn.TruncatedSVD class from sklearn. |
        

--------- 

## Methods 

 

| Function    | Doc             |
|:-------|:----------------|
        | transform | Send the data through all applicable steps to vectorize. |
         
 

### transform

``` python 
    transform(data, fit=False) 
```


Send the data through all applicable steps to vectorize.


# Optimizer 

``` python 
 class Optimizer(object) 
```

Current placeholder for grid methods. Should be fleshed out.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | classifiers | dict, optional, default None |         Dictionary where the key is a initiated model class, and the values        are a dictionary with parameter settings in a (string-array) format,        same as used in the scikit-learn pipeline. So for example, we provide:        {LinearSVC(class_weight='balanced'): {'C': np.logspace(-3, 2, 6)}}.        Note that pipeline requires some namespace (like clf__C), but the class        handles that already. |
        

--------- 

## Methods 

 

| Function    | Doc             |
|:-------|:----------------|
        | best_model | Choose best parameters of trained classifiers. |
        | choose_classifier | Choose a classifier based on settings. |
         
 

### best_model

``` python 
    best_model() 
```


Choose best parameters of trained classifiers.

### choose_classifier

``` python 
    choose_classifier(X, y, seed) 
```


Choose a classifier based on settings.