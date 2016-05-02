# Experiment wrapper code


# Experiment 

``` python 
 class Experiment(conf, cold=False) 
```

Full experiment wrapper.

Calls several sklearn modules in the Pipeline class and reports on the
classifier performance. For this, the class uses a configuration
dictionary. The full list of options for this is listed under attributes.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | cold | boolean, optional, default False | If true, will not immediately run the experiment after calling theclass. Generally we assume that one immediately wants to run on call. |
        

| Attributes    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | "project" | "project_name" |         The project name functions as a hook to for example call the best        performing set of parameters out of a series of experiments on the same        data. |
        | "train_data" | [CSV("/somedir/train.csv", label=1, text=2), |                    CSV("/somedir/train2.csv", label=3, text=5]        The data on which the experiment will train. If the location of a .csv        is provided, it will open these up and create an iterator for you.        Alternatively, you can provide your own iterators or iterable        structures providing instances of the data. If only training data is        provided, the experiment will evaluate in a tenfold setting by default. |
        | "test_proportion" | 0.3 |         As opposed to a test FILE, one can also provide a test proportion,        after which a certain amount of instances will be held out from the        training data to test on. |
        | "backbone" | Spacy() |         The backbone is used as an all-round NLP toolkit for tagging, parsing        and in general annotating the text that is provided to the experiment.        If you wish to utilize features that need for example tokens, lemmas or        POS tags, they can be parsed during loading. Please be advised that        it's more convenient to do this yourself beforehand. |
        | "save" | ("log", model", "db", "man", "json", "pickle") | Save the output of the log, or dump the entire model with itsclassification method and pipeline wrapper for new data instances. |
        

--------- 

## Methods 

 

| Function    | Doc             |
|:-------|:----------------|
        | save | Save desired Experiment data. |
        | run | Split data, fit, transfrom features, tf*idf, svd, report. |
         
 

### save

``` python 
    save() 
```


Save desired Experiment data.

### run

``` python 
    run(conf) 
```


Split data, fit, transfrom features, tf*idf, svd, report.