"""Frog-related methods."""


def decode_frogstring_train(frogstring):
    """Decoder of frogged data in the Amica csv-files.

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
    lines = frogstring.split("\n")
    decoded = []
    for line in lines:
        # add a tuple with (token,lemma,postag,sentence index)
        decoded.append(line.split("\t"))
    return decoded


def process_raw_text(text, lmdir):
    """Extract frog tags.

    Function to convert raw text into an instance list with frog column. Can be
    used for processing new inputs in a demo setting. Returns a list with
    values.

    Parameters
    -----
    text : string
        A raw string of characters.

    Returns
    -----
    instance : list
        The 'row' of one instance, containing all metadata fields that
        are present in the Amica csv-files, as well as the text and
        frogged column.
    """
    import frog
    fo = frog.FrogOptions(parser=False, ner=False)
    frogger = frog.Frog(fo, lmdir + "LaMachine/lamachine/etc/frog/frog-twitter.cfg")
    # add frogged text
    data = frogger.process(text)
    tokens = []
    sentence = -1
    for token in data:
        if token["index"] == '1':
            sentence += 1
        tokens.append([token["text"], token["lemma"],
                       token["pos"], str(sentence)])
    return tokens


def extract_tags(document, tags):
    """
    Extract frog tags.

    Function to extract a list of tags from a frogged document. Document is the
    frogged column of a single document. Tags is a list with any of 'token',
    'lemma', 'postag', or 'sentence' (can just be one of them).

    Parameters
    -----
    document : list of tuples
        Corresponding to all tokens in a single text. A token is a list with
        the fields 'token', 'lemma', 'postag' and 'sentence'.
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
