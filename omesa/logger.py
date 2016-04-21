"""Classes to log and display pipeline information."""

from collections import OrderedDict


class Logger(object):
    """Provides feedback to the user and can store settings in a log file.

    Class holds a log string that can be formatted according to the used
    components and is used to list settings that are provided to the
    experiment. Aside from flat printing, it can iteratively store certain
    lines that are reused (such as loading in multiple datasets). The save
    function makes sure the self.log items are saved according to their logical
    order.

    Parameters
    ----------
    fn: str
        File name of the logfile (and the experiment).

    Attributes
    ----------
    fn: str
        File name.

    log: dict
        Keys are short names for each step, values are strings with .format
        placeholders. Formatting is sort of handled by the strings as well.

    buffer: list
        Used to stack lines in a loop that can be written to the log line once
        the loop has been completed.
    """

    def __init__(self, file_name):
        """Set log dict. Empty buffer."""
        self.file_name = file_name + '.log'
        self.log = {
            'head':   "\n---- Omesa ---- \n\n Config: \n" +
                      "\t {0} \n\tname: {1} \n\tseed: {2} " +
                      "\n\t \n",
            # 'read':   "\n Reading from {0}... Acquired {1} from data.\n ",
            'sparse': "\n Sparse {0} shape: {1}",
            'svd':    "\n Fitting SVD with {0} components...",
            'rep':    "\n\n---- {0} Results ---- \n" +
                      "\n Distribution: {1}" +
                      "\n Accuracy @ baseline: \t {2}" +
                      "\n Reporting on class {3}",
            'grid':   "\n Model with rank: {0} " +
                      "\n Mean validation score: {1:.3f} (std: {2:.3f}) " +
                      "\n Parameters: {3} \n",
            'tfcv':   "\n Tf-CV Result: {0}",
            'f1sc':   "\n F1 Result: {0}",
            'cr':     "\n Performance on test set: \n\n{0}"
        }
        self.buffer = []

    @staticmethod
    def echo(*args):
        """Replacement for a print statement. Legacy function."""
        message = ' '.join([str(x) for x in args])
        print(message)

    def loop(self, key, value):
        """Print and store line to buffer."""
        line = self.log[key].format(*value)
        print(line)
        self.buffer.append(line)

    def dump(self, key):
        """Dump buffer to log."""
        self.log[key] = ''.join(self.buffer)
        self.buffer = []

    def post(self, key, value):
        """Print and store line to log."""
        line = self.log[key].format(*value)
        print(line)
        self.log[key] = line

    def save(self):
        """Save log."""
        with open(self.file_name, 'w') as f:
            o = ['head', 'sparse', 'svd', 'rep', 'grid', 'tfcv', 'f1sc', 'cr']
            f.write(' '.join(
                [v for v in OrderedDict(
                    sorted(self.log.items(), key=lambda i: o.index(i[0]))
                    ).values()]))
