"""Blitzdb database declarations."""

from blitzdb import Document
from blitzdb import FileBackend
from os.path import expanduser
from .tools import serialize_sk as sr


class Configuration(Document):
    """Blitzdb configuration placeholder."""

    class Meta(Document.Meta):
        primary_key = 'name'


class Vectorizer(Document):
    """Blitzdb vectorizer placeholder."""

    class Meta(Document.Meta):
        primary_key = 'name'


class Classifier(Document):
    """Blitzdb classifier placeholder."""

    class Meta(Document.Meta):
        primary_key = 'name'


class Results(Document):
    """Blitzdb results placeholder."""

    class Meta(Document.Meta):
        primary_key = 'name'


class Table(Document):
    """Blitzdb table placeholder."""

    class Meta(Document.Meta):
        primary_key = 'name'


class Database(object):
    """Blitzdb database."""

    def __init__(self):
        """Load backend."""
        self.db = FileBackend(expanduser("~/.omesa/db"))

    def _query(self, f, q):
        try:
            out = f(*q)
        except KeyError:
            self.db = FileBackend(expanduser("~/.omesa/db"))
            f = self.db.filter
            out = f(*q)
        return out

    def save(self, doc):
        """Save document do db."""
        self.db.save(doc)
        self.db.commit()

    def fetch(self, doc, q):
        """Filter and return first entry."""
        try:
            return self._query(self.db.filter, (doc, q))[0]
        except IndexError:
            print("File does not exist.")

    def get_component(self, doc, name):
        # FIXME: see if returning non-decoded is relevant for anything
        try:
            return sr.decode(dict(self._query(
                self.db.filter, (doc, {'name': name}))[0]))
        except IndexError:
            print("File does not exist.")

    def getall(self, doc):
        """Returns all entries in db."""
        return [d for d in self._query(self.db.filter, (doc, {}))]
