"""Blitzdb database declarations."""

from blitzdb import Document
from blitzdb import FileBackend


class Experiment(Document):
    """Blitzdb document placeholder."""

    pass


class Database(object):
    """Blitzdb database."""

    def __init__(self):
        """Load backend."""
        #TODO: I'm sure the path here can be done neater
        self.db = FileBackend(__file__.split('/database.py')[0] + "/db")

    def save(self, doc):
        self.db.save(doc)
        self.db.commit()

    def fetch(self, doc, q):
        try:
            return self.db.filter(doc, q)[0]
        except IndexError:
            print("File does not exist.")
