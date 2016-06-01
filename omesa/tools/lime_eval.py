"""LIME evaluation from Ribeiro, Singh, and Guestrin (2016)."""

import csv

import plotly.offline as py
import plotly.graph_objs as go

from lime.lime_text import ScikitClassifier, LimeTextExplainer


class LimeEval(object):
    """Local linear approximation of a model's behaviour.

    Lime is able to explain any black box text classifier, with two or more
    classes. All it requires is that the classifier implements a function that
    takes in raw text and outputs a probability for each class. Support for
    scikit-learn classifiers is built-in.

    On top of the author's code, this provides several functions to make it
    work with both the Omesa pipeline, as well as the front-end.

    Parameters
    ----------
    cls_ : class
        Scikit-learn classifier.

    vectorizer: class
        Can be either a Scikit-learn classifier or omesa.components.vectorizer.

    Attributes
    ----------
    c : class
        Initialized lime.ScikitlearnClassifier.

    names : array-type of strings
        List (or array) with string versions of labels (to be displayed).

    docs : list of strings
        List of input documents. These are automatically retrieved from the
        configuration["lime_data"] (if set). This process is handled when using
        LimeEval.load_omesa.

    Examples
    --------
    >>> grab = sklearn.datasets.fetch_20newsgroups
    >>> D, Di = grab(subset='train'), grab(subset='test')

    >>> vec = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    >>> X = vec.fit_transform(D.data)
    >>> Xi = vec.transform(Di.data)

    >>> rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    >>> rf.fit(X, D.target)

    >>> import omesa.tools.lime_eval as le
    >>> le.LimeEval(rf, vec, class_names=list(set(Di.target)))
    >>> exps = le.explain(Di[:5])
    >>> graph_to_file(exps, '/some/file/wherever')

    Notes
    -----
    Package from: https://github.com/marcotcr/lime.
    """

    def __init__(self, classifier=None, vectorizer=None, class_names=None,
                 docs=None):
        """Start lime classifier, set label and empty doc placeholder."""
        self.c = ScikitClassifier(classifier, vectorizer)
        self.names = class_names
        self.docs = [] if not docs else docs
        self.multi = 3 if len(class_names) > 2 else None

    def explain(self, docs):
        """Generate LIME Explanations for list of docs.

        Takes as input a list of strings that make up the documents where LIME
        should be applied to. Returns Explanation class instances.

        Parameters
        ----------
        docs : list of strings
            List of input documents.

        Returns
        -------
        exps : list of classes
            For each input document, an Explanation class object on which for
            example the .to_list, to_notebook etc functions can be called on.
        """
        explainer = LimeTextExplainer(class_names=self.names)
        exps = []
        for doc in docs:  # NOTE: this might have messed up in a generator
            exp = explainer.explain_instance(doc, self.c.predict_proba,
                                             top_labels=self.multi)
            exps.append(exp)
        return exps

    def load_omesa(self, lime_repr):
        """Special LIME loader for Omesa pipelines.

        Tries to find a path location to extract example documents from, and
        use the indices set in the omesa.containers object. This requires a
        __dict__ representation of the reader class to have self.path (full
        system path of a file), and self.idx (list of integers with
        [text_index, label_index, etc.]). Advisable to only be used icw Omesa.

        Parameters
        ----------
        lime_repr : ...
            ...

        Returns
        -------
        docs : list of strings
            The top 5 (assuming it has a header) documents from a
            omesa.containers object.
        """
        if isinstance(lime_repr, dict):
            reader = csv.reader(open(lime_repr['path']), quotechar='"')
            ti = lime_repr['idx'][0]
            for i, row in enumerate(reader):
                if lime_repr.get('no_header') and not i:
                    self.docs.append(row[ti])
                elif i:
                    self.docs.append(row[ti])
                if i == 5:
                    break
        else:
            self.docs = lime_repr
        return self.explain(self.docs)

    @staticmethod
    def graph_to_file(exps, loc):
        """Dump LIME experiments with .to_html.

        This is the native way of graphing using LIME, and uses d3.js. The
        files are generally lighter (1 vs 3 MB) than using Omesa (uses plotly).
        However, as they would have to be embedded in the front-end using an
        iframe, the style cannot be changed. As such, this function is ommitted
        when using the Omesa front-end.

        Parameters
        ----------
        exps : list of classes
            An Explanation class for each document.

        loc : str
            The location where to save. If this is used in bottle, should be
            specifically set to `None` to save it to ./static.

        Returns
        -------
        f_names : list of strings
            List of pointers to the file locations where the graphs have been
            stored.
        """
        f_names = []
        for i, exp in enumerate(exps):
            if not loc:
                loc = '/tmp/plot/'
            with open(loc + 'lime_' + str(i) + '.html', 'w') as f:
                html_str = exp.as_html()
                f.write(html_str)
                f_names.append(f.name[1:])
        return f_names

    @staticmethod
    def save_graph(data, layout):
        """Quick binder to dump plotly data and layout."""
        fig = go.Figure(data=data, layout=layout)
        return py.plot(fig, output_type='div', auto_open=False,
                       show_link=False, include_plotlyjs=False)

    def prob_graph(self, i, prob, cln, cols):
        """Output LIME class probability graph. Works with 'graphs' method."""
        data = [go.Bar(x=list(prob), y=cln,
                       marker=dict(color=cols),
                       orientation='h')]
        layout = go.Layout(margin=go.Margin(l=100, r=0, b=0, t=0, pad=0))
        return self.save_graph(data, layout)

    def weight_graph(self, i, expl, cols):
        """Output LIME weight graph. Works with 'graphs' method."""
        # FIXME: colours are binary only
        data = [go.Bar(x=[float(val) for word, val in expl],
                       y=[word for word, val in expl],
                       marker=dict(color=[cols[0] if val < 0 else cols[1]
                                          for word, val in expl]),
                       orientation='h')]
        layout = go.Layout(margin=go.Margin(l=100, r=0, b=0, t=0, pad=0))
        return self.save_graph(data, layout)

    def tag_text(self, i, expl, cols):
        """Highlight LIME top-word in text. Works with 'graphs' method."""
        # FIXME: replace special chars with space and replace on token
        repl = [(word, ('__LIMENEG__' if val < 0 else '__LIMEPOS__') +
                 word + '</span></b>') for word, val in expl]
        doc = str(self.docs[i]).replace('"', '')
        for y in repl:
            doc = doc.replace(*y)
        # these are split up in tokens so that f.e. '1' doesn't screw it up
        doc = doc.replace(
            '__LIMENEG__', '<b><span style="color:{0}">'.format(cols[0]))
        doc = doc.replace(
            '__LIMEPOS__', '<b><span style="color:{0}">'.format(cols[1]))
        return doc

    def unwind(self, exp, comp=False):
        """Unwind LIME experiment in its results."""
        if not comp:
            expl = exp.as_list()
            prb = exp.predict_proba
            cln = exp.class_names
        else:
            expl, prb, cln = exp['expl'], exp['prb'], exp['cln']
        return expl, prb, cln

    def graphs(self, exps, comp=False):
        """Convert exps list to graph locations and annotated text."""
        import colorlover as cl
        order = []
        for i, exp in enumerate(exps):
            expl, prb, cln = self.unwind(exp, comp)
            try:
                ncol = cl.scales[str(len(cln))]['qual']['Set2']
            except KeyError:
                ncol = cl.scales[str(len(cln) + 1)]['qual']['Set2']
            cols = cl.to_rgb(ncol)
            order.append([self.prob_graph(i, prb, cln, cols),
                          self.weight_graph(i, expl, cols),
                          self.tag_text(i, expl, cols)])
        return order

    def to_web(self, tab):
        xps = tab.get('lime_data_comp')
        if not xps:
            xps = self.load_omesa(tab['lime_data_repr'])
        else:
            self.docs = tab.get('lime_data')
        return [x for x in self.graphs(xps, comp=isinstance(xps[0], dict))] \
            if xps else ["Model not probability-based."]
