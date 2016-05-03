"""Main stuff."""

import os, sys
import json
from collections import OrderedDict

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append('../')

import bottle
from omesa.containers import Pipeline
from omesa.database import Database, Table, Results
from sklearn import *
import plotly.offline as py
import plotly.graph_objs as go
import omesa.tools.lime_eval as le
import omesa.tools.serialize_sk as sr


@bottle.route('/static/<filename:path>')
def server_static(filename):
    """Static file includes."""
    return bottle.static_file(filename, root='static')


@bottle.get('/favicon.ico')
def get_favicon():
    """Favicon."""
    return server_static('favicon.ico')


def skeleton(hook='', layout='main', page=''):
    """Skeleton for nesting templates."""
    return bottle.template(
        'main',
        header=bottle.template(layout + '_header'),
        page=page,
        content=hook,
        footer=bottle.template(layout + '_footer')
    )


@bottle.route('/')
def root():
    """Main page."""
    return skeleton(page='Dashboard')


@bottle.route('/run')
def run():
    """Run experiment page."""
    return skeleton(page='Run Experiment', layout='run',
                    hook=bottle.template('run'))


@bottle.route('/exp')
def overview():
    """Experiment overview page."""
    res, out = db.getall(Table), {}
    rows = ['project', 'name', 'train_data', 'test_data', 'features',
            'clf', 'dur', 'test_score']
    out.update({str(xp['pk']): {k: xp[k] for k in rows} for xp in res})
    res = None
    return skeleton(page='Experimental Results', layout='exp',
                    hook=bottle.template('exp', data=out))


def save_graph(tag, data):
    """Quick binder to tag plot, dumps plotly data and layout."""
    layout = go.Layout(margin=go.Margin(l=30, r=30, b=30, t=30, pad=4))
    fig = go.Figure(data=data, layout=layout)
    fn = './static/plots/{0}.html'.format(tag)
    py.plot(fig, filename=fn, auto_open=False, show_link=False)
    return fn[1:]


def test_train_plot(exp):
    tr_score = exp.res['train']['score'] if exp.res.get('train') else 0.0
    te_score = exp.res['test']['score'] if exp.res.get('test') else 0.0
    if not exp.res['prop']:
        data = [
            go.Bar(
                x=['train', 'test'],
                y=[tr_score, te_score]
            )
        ]
    else:
        props, train, test = [], [], []
        d = OrderedDict(sorted(exp.res['prop'].items(), key=lambda t: t[0]))
        for prop, scores in d.items():
            props.append(prop)
            train.append(scores['train'])
            test.append(scores.get('test', 0.0))
        train_trace = go.Scatter(
            x=props + [1.0],
            y=train + [tr_score],
            mode='lines+markers',
            name='train'
        )
        test_trace = go.Scatter(
            x=props + [1.0],
            y=test + [te_score],
            mode='lines+markers',
            name='test'
        )
        data = [train_trace, test_trace]
    return save_graph('basic-bar', data)


def confusion_matrix(t, y_true, y_pred):
    data = [go.Heatmap(z=metrics.confusion_matrix(y_true, y_pred),
                       colorscale=[[0, '#1f77b4'], [1, '#ff7f0e']])]
    return save_graph('heat-' + t, data)


def get_scores(labs, y_true, y_pred):
    # classification report
    p, r, f1, s = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                          average=None,
                                                          labels=labs)
    try:
        acc = metrics.accuracy_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred, average=None)
    except (AttributeError, ValueError):  # mutliclass
        acc, auc = None, None

    scr = []
    for i, label in enumerate(labs):
        scr.append([label] +
                   [round(v, 3) for v in (p[i], r[i], f1[i], s[i])])
    return scr, acc, auc


def unwind_conf(name, tab):
    rows = [('project', 'project name'),
            ('name', 'experiment name'),
            ('train_data_path', 'training data'),
            ('test_data_path', 'testing data'),
            ('features', 'features'),
            ('clf_full', 'classifier'),
            ('dur', 'duration'),
            ('test_score', 'score on test')]
    conf = [(n, tab[k]) for k, n in rows]
    return conf


def lime_eval(exp, tab, labs):
    lime = le.LimeEval(exp.clf, exp.vec, labs)
    if isinstance(tab['lime_data_repr'], dict):
        docs = lime.load_omesa(reader_dict=tab['lime_data_repr'])
    else:
        docs = lime.load_omesa(doc_iter=tab['lime_data_repr'])
    exps = lime.explain(docs)
    return [x for x in lime.graphs(exps)] if exps else \
        ["Model does not support probability prediction and can't do LIME."]


@bottle.route('/exp/<name>')
def experiment(name):
    """Experiment page."""
    exp = Pipeline(name=name, out='db')
    exp.load()

    tab = db.fetch(Table, {'name': name})
    conf = unwind_conf(name, tab)
    # TODO: replace labels with multi-class variant
    labs = exp.vec.encoder.inverse_transform([0, 1])

    lime = lime_eval(exp, tab, labs)
    test_train_plot(exp)

    # heatmap
    scores = sr.decode(json.dumps(dict(db.fetch(Results, {'name': name}))))
    heats, rep = [], []
    for t in ('train', 'test'):
        y_true = exp.vec.encoder.inverse_transform(scores[t]['y'])
        y_pred = exp.vec.encoder.inverse_transform(scores[t]['res'])
        heats.append((t, confusion_matrix(t, y_true, y_pred)))
        scr, acc, auc = get_scores(labs, y_true, y_pred)
        rep.append([t, scr, ('acc', acc), ('auc', auc)])
    return skeleton(page=name, layout='res',
                    hook=bottle.template('res', conf=conf,
                                         plot="/static/plots/basic-bar.html",
                                         lime=lime, heat=heats, rep=rep,
                                         labs=labs))


def main():
    """Main call to app."""
    bottle.debug(True)
    bottle.run(app=bottle.app(), host='localhost', port=6666,
               quiet=False, reloader=True)

if __name__ == '__main__':
    db = Database()
    main()
