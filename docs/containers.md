# Data handling containers


# Pipeline 

``` python 
 class Pipeline(exp=None, name=None, out=None) 
```

Shell for experiment pipeline storing and handling.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | exp | class, optional, default None |         Instance of Experimen with fitted pipes. If not supplied, name and        source should be set. |
        | out | tuple, optional, default None | Instance of Experimen with fitted pipes. If not supplied, name andsource should be set.Tuple with storage options, can be "json" (json serialization),or "db" (for database storage, requires blitzdb). |
        

--------- 

## Methods 

 

| Function    | Doc             |
|:-------|:----------------|
        | _make_tab | Tabular level experiment representation. |
        | save | Save experiment and classifier in format specified. |
        | load | Load experiment and classifier from source specified. |
        | classify | Given a data point, return a (label, probability) tuple. |
         
 

### _make_tab

``` python 
    _make_tab() 
```


Tabular level experiment representation.

Generates a table-level representation of an experiment. This stores
JSON native information ONLY, and is used for the experiment table in
the front-end, as deserializing a lot of experiments will be expensive
in terms of loading times.

### save

``` python 
    save() 
```


Save experiment and classifier in format specified.

### load

``` python 
    load() 
```


Load experiment and classifier from source specified.

### classify

``` python 
    classify(data) 
```


Given a data point, return a (label, probability) tuple.


# CSV 

``` python 
 class CSV 
```

Quick and dirty csv loader.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | text | integer |         Index integer of the .csv where the text is located. |
        | parse | integer, optional, default None |         Index integer of the .csv where the text is located.        Index integer of the .csv where the annotations are provided. Currently        it assumes that these are per instance a list of, for every word,        (token, lemma, POS). Frog and spaCy are implemented to provide these        for you. |
        | header | boolean, optional, default False |         Index integer of the .csv where the text is located.        Index integer of the .csv where the annotations are provided. Currently        it assumes that these are per instance a list of, for every word,        (token, lemma, POS). Frog and spaCy are implemented to provide these        for you.        If the file has a header, you can skip it by setting this to true. |
        

--------- 

## Methods 

 

| Function    | Doc             |
|:-------|:----------------|
        | __iter__ | Standard iter method. |
        | __next__ | Iterate through csv file. |
         
 

### __iter__

``` python 
    __iter__() 
```


Standard iter method.

### __next__

``` python 
    __next__() 
```


Iterate through csv file.