import csv
import plotly.offline as py
import plotly.graph_objs as go
from lime import lime_text
from lime.lime_text import ScikitClassifier, LimeTextExplainer


class LimeEval(object):

    def __init__(self, cls_, vectorizer, class_names=None):
        self.c = ScikitClassifier(cls_, vectorizer)
        self.names = class_names
        self.docs = []

    def explain(self, docs):
        explainer = LimeTextExplainer(class_names=self.names)
        exps = []
        for doc in docs:
            exp = explainer.explain_instance(doc, self.c.predict_proba)
            exps.append(exp)
        return exps

    def load_omesa(self, reader_dict):
        reader = csv.reader(open(reader_dict['path']), quotechar='"')
        ti, docs = reader_dict['idx'][0], []
        for i, row in enumerate(reader):
            if reader_dict.get('header') and not i:
                continue
            docs.append(row[ti])
            if i == 5:
                break
        self.docs = docs
        return docs

    def graph_to_file(self, exps, web=True):
        f_names = []
        for i, exp in enumerate(exps):
            if web:
                loc = './static/'
            with open(loc + 'lime_' + str(i)+'.html', 'w') as f:
                html_str = exp.as_html()
                f.write(html_str)
                f_names.append(f.name[1:])
        return f_names

    def graphs(self, exps, encoder=None):
        order = []
        for i, exp in enumerate(exps):
            expl = exp.as_list()
            prb = exp.predict_proba
            cln = exp.class_names
            graphs = []
            # TODO: clean this up
            # ---
            data = [
                go.Bar(
                    x=list(prb),
                    y=cln,
                    marker=dict(color=['#1f77b4', '#ff7f0e']),
                    orientation='h',
                )
            ]
            fn = './static/lime-prob-{0}.html'.format(i)
            plot_url = py.plot(data, filename=fn, auto_open=False,
                               show_link=False, output_type='file')
            graphs.append(fn[1:])
            # ---
            data = [
                go.Bar(
                    x=[float(val) for word, val in expl],
                    y=[word for word, val in expl],
                    marker=dict(color=['#1f77b4' if val < 0 else '#ff7f0e' for
                                       word, val in expl]),
                    orientation='h',
                )
            ]
            layout = go.Layout(
                xaxis=dict(
                    range=[-1.0, 1.0]
                )
            )
            fig = go.Figure(data=data, layout=layout)
            fn = './static/lime-data-{0}.html'.format(i)
            plot_url = py.plot(fig, filename=fn, auto_open=False,
                               show_link=False, output_type='file')
            graphs.append(fn[1:])
            # ---
            repl = [(' ' + word + ' ', (' <span style="color:#1f77b4">' if
                    val < 0 else ' <span style="color:#ff7f0e">') +
                    word + '</span> ')
                    for word, val in expl]
            doc = str(self.docs[i]).replace('"', '')
            for y in repl:
                doc = doc.replace(*y)
            graphs.append(doc)
            # ----
            order.append(graphs)
        return order
