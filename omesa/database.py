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
        self.db = FileBackend("./db")
