"""Main stuff."""

import os, sys

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append('../')

import bottle
from sklearn import *
import omesa.featurizer
from omesa.containers import Pipeline
from omesa.database import Database, Experiment
import plotly.offline as py
import plotly.graph_objs as go
import omesa.tools.lime_eval as le


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
    plot_html = py.plot(data, filename='./static/basic-bar',
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

    # lime eval
    lime = le.LimeEval(exp.clf, exp.vec)
    docs = lime.load_omesa(res['tab']['lime_data_repr'])
    exps = lime.explain(docs)
    lime = [x for x in lime.graphs(exps)] if exps else \
        ["Model does not support probability prediction and can't do LIME."]

    return skeleton(page=name, layout='res',
                    hook=bottle.template('res', conf=conf,
                                         plot="/static/basic-bar.html",
                                         lime=lime))


def main():
    """Main call to app."""
    bottle.debug(True)
    bottle.run(app=bottle.app(), host='localhost', port=6666,
               quiet=False, reloader=True)

if __name__ == '__main__':
    db = Database()
    main()
