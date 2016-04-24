"""Main stuff."""

import os, sys

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append('../')

import bottle
from omesa.database import Database, Experiment
from omesa.containers import Pipeline
from sklearn import *
import omesa.featurizer

@bottle.route('/static/<filename:path>')
def server_static(filename):
    """Static file includes."""
    return bottle.static_file(filename, root='static')


@bottle.get('/favicon.ico')
def get_favicon():
    """Favicon."""
    return server_static('favicon.ico')


def post_get(name):
    """POST GET."""
    return bottle.request.forms.get(name)


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
            'clf_name', 'dur', 'test_score']
    out.update({str(exp['pk']): {k: exp[k] for k in rows} for exp in res})
    return skeleton(page='Experimental Results', layout='exp',
                    hook=bottle.template('exp', data=out))


@bottle.route('/exp/<name>')
def experiment(name):
    """Experiment page."""
    exp = Pipeline(name=name, source='db')
    x = exp.load()
    return x
    return exp.__dict__


def main():
    """Main call to app."""
    bottle.debug(True)
    bottle.run(app=bottle.app(), host='localhost', port=6666,
               quiet=False, reloader=True)

if __name__ == '__main__':
    db = Database()
    main()
