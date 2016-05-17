# Edit by Chris Emmery, for https://cmry.github.io/notes/serialize.
#
# All credits go to:
#
# Copyright (c) 2013, Christopher R. Wagner
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import namedtuple, Iterable, OrderedDict
import numpy as np
import json
import sys

class Dummy:

    def __init__(self):
        pass

def isnamedtuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return isinstance(obj, tuple) \
           and hasattr(obj, "_fields") \
           and hasattr(obj, "_asdict") \
           and callable(obj._asdict)

def serialize(data):
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [serialize(val) for val in data]
    if isinstance(data, OrderedDict):
        return {"py/collections.OrderedDict":
                [[serialize(k), serialize(v)] for k, v in data.items()]}
    if isnamedtuple(data):
        return {"py/collections.namedtuple": {
            "type":   type(data).__name__,
            "fields": list(data._fields),
            "values": [serialize(getattr(data, f)) for f in data._fields]}}
    # --- custom ---
    if isinstance(data, type):
        return {"py/numpy.type": data.__name__}
    if isinstance(data, np.integer):
        return {"py/numpy.int": int(data)}
    if isinstance(data, np.float):
        return {"py/numpy.float": data.hex()}
    # -------------
    if isinstance(data, dict):
        if all(isinstance(k, str) for k in data):
            return {k: serialize(v) for k, v in data.items()}
        return {"py/dict": [[serialize(k), serialize(v)] for k, v in data.items()]}
    if isinstance(data, tuple):
        return {"py/tuple": [serialize(val) for val in data]}
    if isinstance(data, set):
        return {"py/set": [serialize(val) for val in data]}
    if isinstance(data, np.ndarray):
        return {"py/numpy.ndarray": {
            "values": data.tolist(),
            "dtype":  str(data.dtype)}}
    # --- custom ---
    try:
        _ = data.__next__  # generator as string
        return {'py/generator': str(data)}
    except AttributeError:
        pass
    try:
        if not isinstance(data, type):  # not numpy type
            return {'py/class': {'name': data.__class__.__name__,
                                 'mod': data.__module__,
                                 'attr': serialize(data.__dict__)}}
    except AttributeError as e:
        print(e)
    raise TypeError("Type %s not data-serializable" % type(data))


def restore(dct):
    # --- custom ---
    # print(dct)
    if "py/numpy.type" in dct:
        return np.dtype(dct["py/numpy.type"]).type
    if "py/numpy.int" in dct:
        return np.int32(dct["py/numpy.int"])
    if "py/numpy.float" in dct:
        return np.float64.fromhex(dct["py/numpy.float"])
    # -------------
    if "py/dict" in dct:
        return dict(dct["py/dict"])
    if "py/tuple" in dct:
        return tuple(dct["py/tuple"])
    if "py/set" in dct:
        return set(dct["py/set"])
    if "py/collections.namedtuple" in dct:
        data = dct["py/collections.namedtuple"]
        return namedtuple(data["type"], data["fields"])(*data["values"])
    if "py/numpy.ndarray" in dct:
        data = dct["py/numpy.ndarray"]
        return np.array(data["values"], dtype=data["dtype"])
    if "py/collections.OrderedDict" in dct:
        return OrderedDict(dct["py/collections.OrderedDict"])
    # --- custom ---
    if "py/generator" in dct:
        return []
    if "py/class" in dct:
        obj = dct["py/class"]
        cls_ = getattr(sys.modules[obj['mod']], obj['name'])
        class_init = Dummy()
        class_init.__class__ = cls_
        for k, v in restore(obj['attr']).items():
            setattr(class_init, k, v)
        return class_init
    return dct


def encode(data, fp=False):
    """Python object to file or string."""
    if fp:
        return json.dump(serialize(data), fp)
    else:
        return json.dumps(serialize(data))


def decode(fp):
    """File, String, or Dict to python object."""
    try:
        return json.load(fp, object_hook=restore)
    except (AttributeError, ValueError):
        pass
    try:
        return json.loads(fp, object_hook=restore)
    except (TypeError, ValueError):
        pass
    return restore(fp)
