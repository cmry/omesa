"""Processor methods."""

import re
from os import path

# Author:       Chris Emmery
# Co-authors:   Florian Kunneman, Janneke van de Loo
# License:      BSD 3-Clause
# pylint:       disable=W0142,F0401


class Processor(object):

    """
    Routes to relevant backbone.
    """

    def __init__(self, backbone, wake=False):
        if backbone == 'frog' or (backbone == 'sleepfrog' and wake):
            self.backbone = Frog(path.dirname(path.realpath(__file__)) +
                                 '/../../')
        elif backbone == 'sleepfrog':
            self.backbone = Frog(path.dirname(path.realpath(__file__)) +
                                 '/../../', sleep=True)
        else:
            self.backbone = None
        self.hook = backbone

    def parse(self, text):
        """Route to the parse function of the backbone."""
        return self.backbone.parse(text)

    def decode(self, proc_format_string):
        """Route to the decoding function of the backbone."""
        return self.backbone.decode(proc_format_string)


class Preprocessor(object):

    """
    This is the prerpocessing part.
    """

    def __init__(self):
        self.applied = []

    def basic(self, text):
        text = self.replace_url_email(text)
        text = self.find_emoticons(text)
        text = self.replace_bbcode_tags(text)
        return text

    def replace_bbcode_tags(self, text):
        """
        Replace BBCode tags
        ===================
        Replace all tags with [], which are included in the Netlog data,
        with a tag consisting of capital letters surrounded by underscores.
        Typography tags such as [b], [/b], [u], [/u], [i], [/i] (for bold,
        underlined and italics) are removed.

        The new tags are:
        - _PHOTO_, [photo]116157181[/photo]
        - _VIDEO_, [video]nl-9159440[/video]
        - _URL_, [url=http://www.adres.be/]Adres[/url] or [/url]
        - _EMOTICON_, [love], [@hug], [#clap_anim]

        Parameters
        -----
        text : string
            Input text.

        Returns
        -----
        text : string
            The text in which the Netlog tags with [] have been replaced.
        """
        self.applied.append('repl_bbcode')

        url = re.compile(r"(?:\[url=[^\]]+\][^\[]+)?\[\/url\]")
        photo = re.compile(r"\[photo\][^\]]+\[\/photo\]")
        video = re.compile(r"\[video\][^\[\]]+\[\/video\]")
        typo = re.compile(r"\[\/?[biu]\]")  # bold italics etc.
        emos = re.compile(r"\[[^\[\]]+\]")  # leftovers are probably emots

        text = url.sub('_URL_', text)
        text = photo.sub('_PHOTO_', text)
        text = video.sub('_VIDEO_', text)
        text = typo.sub('', text)
        text = emos.sub('_EMOTICON_', text)
        return text

    def replace_url_email(self, text, repl=('_URL_', '_EMAIL_')):
        """Replace URLs and e-mail addresses.

        Replace URLs with the tag _URL_
        Replace e-mail addresses with the tag _EMAIL_

        Parameters
        -----
        text : string
            Input text.

        Returns
        -----
        text : string
            The text in which the URLs and e-mail addresses have been replaced.
        """
        self.applied.append('repl_url')
        url = re.compile(r"(?:http[s]?://)www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|" +
                         r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        email = re.compile(r"\S+\@[\w\.\-]+\.\w+")
        text = re.sub(url, repl[0], text)
        text = re.sub(email, repl[1], text)
        return text

    def find_emoticons(self, text, repl="_EMOTICON_"):
        """Replace or find emoticons in given text.

        Replace emoticons with a replacement string (default="_EMOTICON_").

        Emoticons can be western (and flipped) -- :), :p, :(, o:, x: -- or
        eastern ^_^.

        Parameters
        -----
        text : string
            Input text.

        repl_str : string, default "_EMOTICON_"
            String with which emoticons are replaced.

        Returns
        -----
        re.sub : string
            The text with the emoticons replaced by repl.
        re.findall : list
            If there is no specified replacement tag, will just return list of
            emoticons.
        """
        if repl:
            self.applied.append('repl_emoticons')
        emoticon_string = r"""
        (?:
          [<>]?
          [:;=8xX]                         # eyes L
          [\-o\*\']?                       # nose L
          [\)\]\(\[dDpPxXoOsS/\:\}\{@\|\\] # mouth L
          |
          [\)\]\(\[dDpPxXoOsS/\:\}\{@\|\\] # mouth R
          [\-o\*\']?                       # nose R
          [:;=8xX]                         # eyes R
          [<>]?
          |
          [\~<>]?                          # hands
          [\(] ?                           # body L
          [\-Oo~\^]                        # eyes L
          [\_\-\.]                         # mouth
          [\-Oo~\^]                        # eyes R
          [\)] ?                           # body R
          [\~<>]?                          # hands
        )"""
        pat = re.compile(emoticon_string, re.VERBOSE | re.I | re.UNICODE)
        return re.sub(pat, repl, text) if repl else re.findall(pat, text)


class Frog(object):

    """
    Wrapper to python-frog, loaded from LaMachine.
    """

    def __init__(self, lmdir, sleep=False):
        """Starts the frog server if the sleep function isn't on."""
        if not sleep:
            import frog
            opts = frog.FrogOptions(parser=False, ner=False)
            self.frogger = frog.Frog(opts, lmdir + "LaMachine/lamachine/etc/"
                                     "frog/frog-twitter.cfg")

    def parse(self, text):
        """Extract frog tags.

        Function to convert raw text into an instance list with frog column.
        Can be used for processing new inputs in a demo setting. Returns a list
        with values.

        Parameters
        -----
        text : string
            A raw string of characters.

        Returns
        -----
        instance : list
            The 'row' of one instance, containing all metadata fields that
            are present in the shed csv-files, as well as the text and
            frogged column.
        """
        # add frogged text
        data = self.frogger.process(text)
        tokens, sentence = [], -1
        for token in data:
            if token["index"] == '1':
                sentence += 1
            tokens.append([token["text"], token["lemma"],
                           token["pos"], str(sentence)])
        return tokens

    @staticmethod
    def decode(frogstring):
        """Decoder of frogged data in the shed csv-files.

        Converts frogged lines into a list with tokens as
        [token, lemma, postag, sentence].

        Parameters
        -----
        frogstring : string
            The frogged data of a single document.

        Returns
        -----
        decoded : list
            The frogstring as list of lists.
        """
        return [line.split("\n") for line in frogstring.split("\n")]

    @staticmethod
    def extract_tags(document, tags):
        """
        Extract frog tags.

        Function to extract a list of tags from a frogged document. Document is
        the frogged column of a single document. Tags is a list with any of
        'token', 'lemma', 'postag', or 'sentence' (can just be one of them).

        Parameters
        -----
        document : list of tuples
            Corresponding to all tokens in a single text. A token is a list
            with the fields 'token', 'lemma', 'postag' and 'sentence'.
        tags: list
            List of tags to return from a document. Options:
            - token
            - lemma
            - postag
            - sentence

        Returns
        -----
        extracted_tags : list
            Sequence of the tokens in the document, with the selected tags
            if the value of 'tags' is '[token, postag]', the output list will
            be '[[token, postag], [token, postag], ...]'.
        """
        tagdict = {'token': 0, 'lemma': 1, 'postag': 2, 'sentence': 3}
        taglists = [[token[tagdict[tag]] for token in document]
                    for tag in tags]
        if len(taglists) > 1:
            return zip(*[taglists])
        else:
            return taglists[0]
