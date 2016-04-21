"""Main stuff."""

import os, sys

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append('../')

import bottle
from database import Database, Experiment


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


@bottle.route('/exp')
def experiment():
    """Experiment page."""
    return skeleton(page='Experiment', layout='exp',
                    hook=bottle.template('exp'))


@bottle.route('/outp')
def results():
    """Experiment page."""
    return skeleton(page='Results', layout='outp',
                    hook=bottle.template('outp',
                        data=db.getall(Experiment)))


def main():
    """Main call to app."""
    bottle.debug(True)
    bottle.run(app=bottle.app(), host='localhost', port=6666,
               quiet=False, reloader=True)

if __name__ == '__main__':
    db = Database()
    main()
