# Namespace Environment

The idea of this module is to provide an interface for loading data, training
and testing models, storing well performing model versions with their
associated data and feature combinations, and the ability to load these all
back in again to test on new data.

``` python
class Environment(name, backbone='frog')
```

Starts the framework's environment and initiates its namespace.


| Parameters   |         |                                                                        |
|: ----------- |: ------ |: --------------------------------------------------------------------- |
|**name**      |  string | The namespace under which you want to save the existing configuration. |

| Attributes   |         |                                                                        |
|: ----------- |: ------ |: --------------------------------------------------------------------- |
|**reader**    | class   | [Reader](datareader.md) located in `datareader.py`                     |
|**featurizer**| class   | [Featurizer](featurizer.md) located in `featurizer.py`                 |
|**model**     | class   | Sklearn / any other external class.                                    |

---

## Examples

Typical experiment:

```python
>>> import shed
>>> from os import getcwd

>>> data = [getcwd()+'/data/data.csv', getcwd()+'/data/data2.csv']

>>> from shed.featurizer import *
>>> features = [SimpleStats(), Ngrams(level='pos'), FuncWords()]

>>> env = shed.Environment(name='bayes_age_v1')
>>> loader = env.load(data=data, target_label='age')
>>> space, labels = env.transform(loader(), features)
```

---

## Methods

| Function                         | Doc
|: ------------------------------- |: ------------------------------------------------- |
| [load](#methods-load)            | Wrapper for the data loader.                       |
| [transform](#methods-transform)  | Transform the data according to required features. |
| [train](#methods-train)          | Fit a sklearn syntax compatible classifier.        |
| [test](#methods-test)            | Test a sklearn syntax compatible classifier.       |
| [classify](#methods-classify)    | Quickly transform and predict a text instance.     |
| [save](#methods-save)            | Save the environment to a folder in model.         |

### load

``` python
load(data, proc=None, max_n=None, skip=range(0, 0), rnd_seed=111,
     label='label', meta=[])
```

Load provided dataset(s) taking into account specified options.

| Param        | Type                                                                                  | Doc                                                  |
|: ----------- |: ------------------------------------------------------------------------------------ |:---------------------------------------------------- |
|**data**      | list of strings                                                                       | List with document directories to be loaded.         |
|**proc**      | string or function | **[None (default), 'text', 'label', 'both', function]** <br>  Any label or text conversion can be indicated with a string, or an own function can be supplied to apply to the row object. |
|**max_n**     | int, optional, default False                                                          | Maximum number of data instances *per dataset* user wants to work with. |
|**skip**      | range, optional, default False                                                        | Range of indices that need to be skipped in the data. Can be used for splitting as in tenfold cross-validation, whilst retaining the iterative functionality and therefore keeping memory consumption low. |
|**rnd_seed**  | int, optional, default 111                                                            | A seed number used for reproducing the random order. |
|**label**     | str, optional, default label                                                          | Name of the label header row that should be retrieved. If not set, the second column will be asummed to be a label column. |
|**meta**      | list, optional, default None                                                          | Extract features from the dataset itself by specifying the headers or the indices in which these are located. Include 'file' if the filename has to be a feature. |

### transform

``` python
transform(loader, features=False)
```

Transform the data according to required features into a sparse representation
in dictionary format.

| Param        | Type                                | Doc                                    |
|: ----------- |: ---------------------------------- |:-------------------------------------- |
| **loader**  | generator                           | The loader should iteratively yield a preprocessed testing data instance with (label, raw, parse, meta). |
| **Returns**  |                                     |                                         |
| **space**    | list of dicts with {feature: value} of shape [n_instances] | Dict with sparse feature representation.  |
| **labels**   | list of shape [n_labels]            | List of labels for data instances.        |


### train

``` python
train(model, space, labels)
```

Fit an sklearn syntax compatible classifier.


| Param        | Type                                | Doc                                    |
|: ----------- |: ---------------------------------- |:-------------------------------------- |
| **model**    | class | Should have a fit method for training on space and labels. |
| **space**    | matrix of shape [n_instances, n_features], or scipy.sparse |  Depending on what model can train on, provide feature matrix. |
| **labels**   | list of shape [n_instances] | List of labels for every row in space. |
### test
