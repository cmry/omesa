"""Processor methods."""

import re
from os import path

# Author:       Chris Emmery
# Contributors: Florian Kunneman
# License:      BSD 3-Clause
# pylint:       disable=W0142,F0401
# -*- coding: utf-8 -*-


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
        elif backbone == 'spacy':
            self.backbone = Spacy()
        else:
            self.backbone = None
        self.hook = backbone

    def parse(self, text):
        """Route to the parse function of the backbone."""
        return self.backbone.parse(text)


class Preprocessor(object):

    """
    Preprocess raw data.

    This class should handle all the preprocessing done for both the labels
    as well as the texts that are provided. Current implementations are the
    standard replacement of bbcode, emoticons and urls with their own
    __TOKENS__. These are placed in the `basic` preprocessing.
    """

    def __init__(self):
        self.applied = []

    def basic(self, text):
        text = self.replace_url_email(text)
        text = self.find_emoticons(text)
        text = self.replace_bbcode_tags(text)
        return text

    def label_convert(self, config, label):
        """
        Label converter.

        Converts a label according to the convert_config as many times as
        specified.

        Parameters
        ----------
        config: dict
            Each key is a label, and each value a tuple with (count, convertto)
            so that label will be convertto as many times as count.

        label : string
            The data entry of the label, to be converted.

        Returns
        -------
        converted_label : string
            The converted label.
        """
        if self.conf[label][0] != 0:
            self.conf[label][0] -= 1
            if len(self.conf[label]) > 1:
                return self.conf[label][1]
            else:
                return label

    def replace_bbcode_tags(self, text):
        """Replace BBCode tags.

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


class Spacy(object):

    """
    Wrapper to spaCy.io. From their docs @ http://http://spacy.io/docs/

    "spaCy consists of a vocabulary table that stores lexical types, a
    pipeline that produce annotations, and three classes to manipulate
    document, span and token data. The annotations are predicted using
    statistical models, according to specifications that follow common
    practice in the research community."

    spaCy is currently used in shed to provide the English part of the
    backbone. It's faster than CoreNLP, and Python <3. While spaCy can also
    extract things such as NER (it lacks sentiment and co-reference), this
    is currently not enabled for shed.
    """

    def __init__(self, lmdir):
        """Load up the spaCy pipeline."""
        from spacy.en import English
        self.spacy = English()

    def parse(self, text):
        """Extract spaCy tags.

        Convert raw text instance into spaCy format. Currently only returns
        token, lemma, POS.

        Parameters
        -----
        text : string
            A raw string of characters.

        Returns
        -----
        instance : list
            The token, lemma, POS list that can be used in featurizers.
        """
        doc = self.spacy(text, tag=True, parse=True)
        return [[token.orth_, token.lemma_, token.pos_] for token in doc]


class Frog(object):

    """
    Wrapper to python-frog, loaded from LaMachine. Excerpt from the
    documentation @ http://ilk.uvt.nl/frog/:

    Frog is an integration of memory-based natural language processing (NLP)
    modules developed for Dutch. All NLP modules are based on Timbl, the
    Tilburg memory-based learning software package. Recently, a dependency
    parser, a base phrase chunker, and a named-entity recognizer module were
    added. Where possible, Frog makes use of multi-processor support to run
    subtasks in parallel.

    Frog is currently used in shed to provide the Dutch part of the backbone.
    As the other backbones, it currently only uses a subset of features. Full
    list of potential extractions (not enabled for shed) are:

    - Morphological segmentation (according to MBMA).
    - Confidence in the POS tag, a number between 0 and 1, representing the
      probability mass assigned to the best guess tag in the tag distribution.
    - Named entity type, identifying person (PER), organization (ORG), location
      (LOC), product (PRO), event (EVE), and miscellaneous (MISC), using a BIO
      (or IOB2) encoding.
    - Base (non-embedded) phrase chunk in BIO encoding.
    - Token number of head word in dependency graph (according to CSI-DP).
    - Type of dependency relation with head word.
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

        Convert raw text instance into Frog format. Currently only returns
        token, lemma, POS.

        Parameters
        -----
        text : string
            A raw string of characters.

        Returns
        -----
        instance : list
            The token, lemma, POS list that can be used in featurizers.
        """
        doc = self.frogger.process(text)
        return [[token["text"], token["lemma"], token["pos"]] for token in doc]
