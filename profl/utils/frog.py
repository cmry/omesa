# import frogger


def decode_frogstring_train(self, frogstring):
    lines = frogstring.split("\n")
    decoded = []
    for line in lines:
        # add a tuple with (token,lemma,postag,sentence index)
        decoded.append(line.split("\t"))
    return decoded


def process_raw_text(self, text):
    """
    function to convert raw text into an instance list with frog column
    can be used for processing new inputs in a demo setting
    returns a list with values
    """
    # initialize list
    instance = [False] * 9  # empty fields
    instance.append(text)
    # add frogged text
    fr = frogger.Frogger([text])
    fr.frog_docs()
    fr.decode_frogstrings()
    instance.append(fr.decoded_frogstrings[0])
    return instance


def extract_tags(self, document, tags):
    """
    Function to extract a list of tags from a frogged document
    Document is the frogged column of a single instance
    Tags is a list with any of 'token', 'lemma', 'postag', or 'sentence'
    (can just be one of them)
    """
    tagdict = {'token': 0, 'lemma': 1, 'postag': 2, 'sentence': 3}
    taglists = [[token[tagdict[tag]] for token in document]
                for tag in tags]
    if len(taglists) > 1:
        return zip(*[taglists])
    else:
        return taglists[0]
