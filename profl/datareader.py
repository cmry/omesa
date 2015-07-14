"""Data handling functions.

Used to open a stream, and activate certain preprocessing options such as
label conversion.
"""
import random as rnd
import sys
import csv

from .utils import frog

# Author:       Chris Emmery
# Co-author:    Florian Kunneman
# License:      BSD 3-Clause
# pylint:       disable=E1103


class Datareader:

    r"""
    Classification data handler.

    Can convert headed .csv files to an instance based streaming of the raw
    data and the specified label in the header. Given that frog is installed,
    it can also convert a new untagged instance to a tagged format.

    Parameters
    ----------
    data : list of strings
        List with document directories to be loaded.

    proc : string or function, [None (default), 'text', 'label', 'both', \
                                function]
        If you want any label or text conversion, you can indicate this
        here with a string, or either supply your own to apply to the row
        object. These are constructed as list[label, text, frog].

        'text':
            Apply a generic normalization and preprocessing process to the
            input data.
        'label':
            Do a general categorization based on the labels that is
            provided if need be. Age will for example be splitted into
            several clases.
        'both':
            All of the above.
        function:
            Specify your own function by which you want to edit the row.

    max_n : int, optional, default False
        Maximum number of data instances *per dataset* user wants to work
        with.

    shuffle : bool, optional, default True
        If the order of the dataset should be randomized.

    rnd_seed : int, optional, default 666
        A seed number used for reproducing the random order.

    label : str, optional, default age header
        Name of the label header row that should be retrieved. If not set,
        the second column will be asummed to be a label column.

    meta : list of str, optional, default empty
        If you'd like to extract features from the dataset itself, this can
        be used to specify the headers or the indices in which these are
        located. Include 'file' if you want the filename to be a feature.

    Attributes
    ----------
    datasets : dict
        Dictionary where key is the name of a dataset, and the value its
        rows. Will not be filled if data is streamed.

    headers : list
        List of headers / labels if they were found present in the
        datasets. If not, standard AMiCA list is provided. Might
        introduce bugs.

    Examples
    --------
    >>> data = ['~/Documents/data1.csv', '~/Documents/data2.csv']
    >>> reader = Datareader(data, max_n=1)
    >>> labels, raw, frog = zip(*reader.load(data))
    >>> pp(labels, raw, frog)
    (('38',),
     ("'lieve schat we kunnen miss beter wel eentje gaan drinken samen xxx'",),
     ([["'", "'", 'LET()', '0'],
       ['lieve', 'lief', 'ADJ(prenom,basis,met-e,stan)', '0'],
       ['schat', 'schat', 'N(soort,ev,basis,zijd,stan)', '0'],
       ['we', 'we', 'VNW(pers,pron,nomin,red,1,mv)', '0'],
       ['kunnen', 'kunnen', 'WW(pv,tgw,mv)', '0'],
       ['miss', 'miss', 'SPEC(vreemd)', '0'],
       ['beter', 'goed', 'ADJ(vrij,comp,zonder)', '0'],
       ['wel', 'wel', 'BW()', '0'],
       ['eentje', 'een', 'TW(hoofd,nom,zonder-n,dim)', '0'],
       ['gaan', 'gaan', 'WW(inf,vrij,zonder)', '0'],
       ['drinken', 'drinken', 'WW(inf,vrij,zonder)', '0'],
       ['samen', 'samen', 'BW()', '0'],
       ['xxx', 'xxx', 'SPEC(vreemd)', '0'],
       ["'", "'", 'LET()', '0']],))

    Notes
    -----
    Interactive use has been deprecated in this version. Changed its scope in
    the pipeline to public.
    """

    def __init__(self, data, proc='both', max_n=None, shuffle=True,
                 rnd_seed=666, label='age', meta=[]):
        """Initialize the reader with restrictive parameters."""
        self.file_list = data
        self.proc = proc
        self.max_n = max_n+1 if max_n else None  # due to offset
        self.shuffle = shuffle
        self.rnd_seed = rnd_seed
        self.label = label
        self.headers = "user_id age gender loc_country loc_region \
                       loc_city education pers_big5 pers_mbti text \
                       frog".split()
        self.meta = meta
        self.datasets = {}

        rnd.seed(self.rnd_seed)

    def load(self):
        """
        Raw data generator.

        This is now the main way to load in your .csv files. It will check
        which labels are present in the input data, and will isolate any
        specified label. Please note that frog data **must** be under a 'frogs'
        header, otherwise it will try to retag it as new data!

        Yields
        ------
        p_row : list
            Has the following objects from 0-3:

            labels : list
                Labels for each of the instances in raw, labels should be
                maintained in the same position for e.g. sklearn evaluation.
            raw : list
                The raw data comes in an array where each entry represents a
                text instance in the data file.
            frogs : list
                The frog data, list is empty if no data is found.
            meta : list
                A list of meta-information features if these were specified.
        """
        for file_name in self.file_list:
            for row in self._load_data_linewise(file_name):
                p_row = self._preprocess(row)
                if p_row:
                    yield p_row

    def _label_convert(self, label_field):
        """
        Label converter.

        Converts whatever field it is given to some format specified according
        to the self.label.

        Parameters
        ----------
        label_field : string
            The data entry of the label, to be converted.

        Returns
        -------
        converted_label : string
            The converted label.
        """
        if self.label == 'age':
            age = {range(15):      'child',
                   range(15, 18):  'teen',
                   range(18, 21):  'post-teen',
                   range(21, 26):  'young adult',
                   range(26, 100): 'adult'}
            for r in age.keys():
                if int(label_field) in r:
                    return age[r]
        else:
            return label_field

    def _preprocess(self, row):
        """
        Text and label preprocessor.

        Sends the input to either the label processor, or, for now, only does
        text.lower() as text preprocessing.

        Parameters
        ----------
        row : list
            Row pruned to include only the selected label, raw data and decoded
            frog data.

        Returns
        -------
        row : list
            Either preprocessed raw, a converted label or both.
        """
        if self.proc == 'label' or self.proc == 'both':
            new_label = self._label_convert(row[0])
            row[0] = new_label
        if self.proc == 'text' or self.proc == 'both':
            new_text = row[1].lower()
            row[1] = new_text
        if self.proc and not isinstance(self.proc, str):
            row = self.proc(row)
        if None not in row and len(row[0]) > 0 and len(row[2]) > 3:
            return row

    def _extract_row(self, line, filename):
        """
        Data extractor.

        Fetches required data from data files. Handles frog data correctly.

        Parameters
        ----------
        line : list
            List with .csv frow.

        filename : str
            Directory of a .csv file to be stored as a feature.

        Returns
        -------
        rows : list
            Row pruned to include only the selected label, raw data and decoded
            frog data.

        """
        label_data = line[self.headers.index(self.label)]
        text_data = line[self.headers.index('text')]
        meta_data = []
        if 'file' in self.meta:
            self.meta.remove('file')
            meta_data += [filename]
        meta_data += [line[self.headers.index(x)] for x in self.meta]
        try:
            head = self.headers.index('frogs')
            frog_data = frog.decode_frogstring_train(line[head])
        except ValueError:
            frog_data = []
        row = [label_data, text_data, frog_data, meta_data]
        return row

    def _check_header(self, filename):
        """Sniff if a .csv file has a header."""
        with open(filename, 'r') as sniff_file:
            if csv.Sniffer().has_header(sniff_file.read(200)):
                return True

    def _load_data_linewise(self, filename):
        """
        Csv reader.

        Reads a csv file by pathname, extracts headers and returns matrix.

        Parameters
        -----
        filename : str
            Directory of a .csv file to be stored into a list.

        Yields
        -----
        rows : list
            List of lists where each row is an instance and column a label
            entry or text data.

        """
        csv.field_size_limit(sys.maxsize)
        with open(filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for i, line in enumerate(csv_reader):
                if self._check_header(filename) and not i:
                    self.headers = line
                    if not self.label:
                        self.label = self.headers[1]
                elif self.max_n and i >= self.max_n:
                    break
                else:
                    row = self._extract_row(line, filename)
                    if row[0]:  # if label
                        yield row
