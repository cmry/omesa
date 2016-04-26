"""Main stuff."""

import os, sys
import json

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append('../')

import bottle
from omesa.containers import Pipeline
from omesa.database import Database, Experiment
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
    res, out = db.getall(Experiment), {}
    rows = ['project', 'name', 'train_data', 'test_data', 'features',
            'clf', 'dur', 'test_score']
    out.update({str(xp['pk']): {k: xp['tab'][k] for k in rows} for xp in res})
    return skeleton(page='Experimental Results', layout='exp',
                    hook=bottle.template('exp', data=out))


@bottle.route('/exp/<name>')
def experiment(name):
    """Experiment page."""
    exp = Pipeline(name=name, source='db')
    exp.load()

    # test/train plot
    data = [
        go.Bar(
            x=['train', 'test'],
            y=[exp.res['train']['score'] if exp.res.get('train') else 0.0,
               exp.res['test']['score'] if exp.res.get('test') else 0.0]
        )
    ]
    layout = go.Layout(margin=go.Margin(l=30, r=30, b=30, t=30, pad=4))
    fig = go.Figure(data=data, layout=layout)
    plot_html = py.plot(fig, filename='./static/basic-bar.html',
                        auto_open=False, show_link=False, output_type='file')

    # unwind config
    res, conf = db.fetch(Experiment, {'name': name}), []
    rows = [('project', 'project name'),
            ('name', 'experiment name'),
            ('train_data_path', 'training data'),
            ('test_data_path', 'testing data'),
            ('features', 'features'),
            ('clf_full', 'classifier'),
            ('dur', 'duration'),
            ('test_score', 'score on test')]
    conf = [(n, res['tab'][k]) for k, n in rows]

    # TODO: replace labels with multi-class variant
    labs = exp.vec.encoder.inverse_transform([0, 1])
    # lime eval
    lime = le.LimeEval(exp.clf, exp.vec, labs)
    docs = lime.load_omesa(res['tab']['lime_data_repr'])
    exps = lime.explain(docs)
    lime = [x for x in lime.graphs(exps)] if exps else \
        ["Model does not support probability prediction and can't do LIME."]

    # heatmap
    scores = sr.decode(json.dumps(res['res']))
    heats, rep = [], []
    for t in ('train', 'test'):
        y_true = exp.vec.encoder.inverse_transform(scores[t]['y'])
        y_pred = exp.vec.encoder.inverse_transform(scores[t]['res'])
        data = [go.Heatmap(z=metrics.confusion_matrix(y_true, y_pred),
                           colorscale=[[0, '#1f77b4'], [1, '#ff7f0e']])]
        fig = go.Figure(data=data, layout=layout)
        plot_url = py.plot(fig, filename='./static/heat-' + t + '.html',
                           auto_open=False, show_link=False,
                           output_type='file')
        heats.append((t, '/static/heat-' + t + '.html'))

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
        rep.append([t, scr, ('acc', acc), ('auc', auc)])

    return skeleton(page=name, layout='res',
                    hook=bottle.template('res', conf=conf,
                                         plot="/static/basic-bar.html",
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
