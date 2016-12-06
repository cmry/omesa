"""LIME evaluation from Ribeiro, Singh, and Guestrin (2016)."""

import csv

import plotly.offline as py
import plotly.graph_objs as go

from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


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

    def __init__(self, classifier=None, vectorizer=None, n_classes=None, docs=None):
        """Start lime classifier, set label and empty doc placeholder."""
        self.pipeline = make_pipeline(vectorizer, classifier)
        self.documents = [] if not docs else docs
        self.n_classes = n_classes

        # NOTE: given that Python uses a lot of these symbolic links, we might
        # heavily reduce the size and time required for serializing objects if
        # we hash them first, store the hashes, and remember their position.
        # After, we can just refer back and copy this info to extract again.

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
        explainer = LimeTextExplainer()
        experiments = []

        for doc in docs:  # NOTE: this might have messed up in a generator
            experiment = explainer.explain_instance(
                doc, self.pipeline.predict_proba, top_labels=self.n_classes)
            experiments.append(experiment)

        return experiments

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
            text_index = lime_repr['idx'][0]

            for i, row in enumerate(reader):
                if lime_repr.get('no_header') and not i:
                    self.documents.append(row[text_index])
                elif i:
                    self.documents.append(row[text_index])
                if i == 5:
                    break
        else:
            self.documents = lime_repr

        return self.explain(self.documents)

    @staticmethod
    def graph_to_file(explanations, file_dir):
        """Dump LIME experiments with .to_html.

        This is the native way of graphing using LIME, and uses d3.js. The
        files are generally lighter (1 vs 3 MB) than using Omesa (uses plotly).
        However, as they would have to be embedded in the front-end using an
        iframe, the style cannot be changed. As such, this function is ommitted
        when using the Omesa front-end.

        Parameters
        ----------
        explanations : list of classes
            An Explanation class for each document.

        file_dir : str
            The location where to save. If this is used in bottle, should be
            specifically set to `None` to save it to ./static.

        Returns
        -------
        file_names : list of strings
            List of pointers to the file locations where the graphs have been
            stored.
        """
        file_names = []

        for i, explanation in enumerate(explanations):
            if not file_dir:
                file_dir = '/tmp/plot/'

            with open(file_dir + 'lime_' + str(i) + '.html', 'w') as f:
                html_str = explanation.as_html()
                f.write(html_str)
                file_names.append(f.name[1:])

        return file_names

    @staticmethod
    def save_graph(data, layout):
        """Quick binder to dump plotly data and layout."""
        fig = go.Figure(data=data, layout=layout)

        return py.plot(fig, output_type='div', auto_open=False,
                       show_link=False, include_plotlyjs=False)

    def prob_graph(self, i, probabilities, class_names, colors):
        """Output LIME class probability graph. Works with 'graphs' method."""
        data = [go.Bar(
                    x=probabilities,
                    y=class_names,
                    marker=dict(color=colors),
                    orientation='h'
                )]
        layout = go.Layout(margin=go.Margin(l=100, r=0, b=0, t=0, pad=0))

        return self.save_graph(data, layout)

    def weight_graph(self, i, explanations, colors):
        """Output LIME weight graph. Works with 'graphs' method."""
        # FIXME: colours are binary only
        zero_color = 'rgb(128, 128, 128)' if len(colors) > 2 else colors[-1]
        data = [go.Bar(
                    x=[float(weight) for word, weight in explanations],
                    y=[word for word, weight in explanations],
                    marker=dict(color=[
                        zero_color if val < 0 else colors[0]
                                       for word, val in explanations]),
                    orientation='h'
                )]
        layout = go.Layout(margin=go.Margin(l=100, r=0, b=0, t=0, pad=0))

        return self.save_graph(data, layout)

    def tag_text(self, i, explanations, colors):
        """Highlight LIME top-word in text. Works with 'graphs' method."""
        # FIXME: replace special chars with space and replace on token
        zero_color = 'rgb(128, 128, 128)' if len(colors) > 2 else colors[-1]
        replacements = {word: ('<b><span style="color:{0}">'.format(
                                zero_color if val < 0 else colors[0]) + word +
                              '</span></b>') for word, val in explanations}

        new_doc = str(self.documents[i]).replace('"', '').split(' ')
        for idx, word in enumerate(new_doc):
            if word in replacements:
                new_doc[idx] = replacements[word]

        return ' '.join(new_doc)

    def unwind(self, experiment, pre_computed=False):
        """Unwind LIME experiment in its results."""
        if not pre_computed:
            proba = experiment.predict_proba
            class_names = experiment.class_names
            class_names, proba = \
                zip(*[(c, p) for (p, c) in
                      sorted(zip(proba, class_names), reverse=True)])

            explanations = experiment.as_list(label=int(class_names[0]))

        else:
            explanations, proba, class_names = \
                experiment['expl'], experiment['prb'], experiment['cln']

        return explanations, proba, class_names

    def graphs(self, explanations, pre_computed=False):
        """Convert exps list to graph locations and annotated text."""
        import colorlover as cl
        order, color_order = [], {}

        for i, explanation in enumerate(explanations):
            explanation, probabilities, class_names = \
                self.unwind(explanation, pre_computed)

            if not color_order:
                try:
                    n_colors = cl.scales[str(len(class_names))]['qual']['Set2']
                except KeyError:
                    corrected_n = len(class_names) + 1
                    n_colors = cl.scales[str(corrected_n)]['qual']['Set2']

                lime_colors = cl.to_rgb(n_colors)
                color_order = {name: color for color, name in
                               zip(lime_colors, class_names)}

            colors = [color_order[name] for name in class_names]
            order.append([
                self.prob_graph(i, probabilities, class_names, colors),
                self.weight_graph(i, explanation, colors),
                self.tag_text(i, explanation, colors)])

        return order

    def to_web(self, table_data):
        """Dump table information to a web-compatible format."""
        lime_experiments = table_data.get('lime_data_comp')

        if not lime_experiments:
            lime_experiments = self.load_omesa(table_data['lime_data_repr'])
        else:
            # FIXME: this doesn't work anymore?
            self.documents = table_data.get('lime_data')

        if lime_experiments:
            return [graph for graph in
                    self.graphs(lime_experiments,
                            pre_computed=isinstance(lime_experiments[0], dict))
                ]
        else:
            return [["Model not probability-based."]]
