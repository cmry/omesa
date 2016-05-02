# Processor methods


# SimpleCleaner

``` python
 class SimpleCleaner def __init__(self)
```

Very simple data cleaner.

---------

## Methods



| Function    | Doc             |
|:-------|:----------------|
        | clean | Lower text and removes special stuff. |



### clean

``` python
    clean(text)
```


Lower text and removes special stuff.


# SocialCleaner

``` python
 class SocialCleaner(object)
```

Preprocess raw social media data.

This class should handle all the preprocessing done for both the labels
as well as the texts that are provided. Current implementations are the
standard replacement of bbcode, emoticons and urls with their own
__TOKENS__. These are placed in the `basic` preprocessing.

---------

## Methods



| Function    | Doc             |
|:-------|:----------------|
        | clean | Clean according to ALL the preprocessors. |
        | replace_bbcode_tags | Replace BBCode tags. |
        | replace_url_email | Replace URLs and e-mail addresses. |
        | find_emoticons | Replace or find emoticons in given text. |



### clean

``` python
    clean(text)
```


Clean according to ALL the preprocessors.

### replace_bbcode_tags

``` python
    replace_bbcode_tags(text)
```


Replace BBCode tags.

Replace all tags with [], which are included in the Netlog data,
with a tag consisting of capital letters surrounded by underscores.
Typography tags such as [b], [/b], [u], [/u], [i], [/i] (for bold,
underlined and italics) are removed.
The new tags are:
- _PHOTO_, [photo]116157181[/photo]
- _VIDEO_, [video]nl-9159440[/video]
- _URL_, [url=http://www.adres.be/]Adres[/url] or [/url]
- _EMOTICON_, [love], [@hug], [#clap_anim]

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | text | string | Input text. |


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | text | string | The text in which the Netlog tags with [] have been replaced. |


### replace_url_email

``` python
    replace_url_email(text, repl=('_URL_', '_EMAIL_'))
```


Replace URLs and e-mail addresses.

Replace URLs with the tag _URL_
Replace e-mail addresses with the tag _EMAIL_

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | text | string | Input text. |


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | text | string | The text in which the URLs and e-mail addresses have been replaced. |


### find_emoticons

``` python
    find_emoticons(text, repl="_EMOTICON_")
```


Replace or find emoticons in given text.

Replace emoticons with a replacement string (default="_EMOTICON_").
Emoticons can be western (and flipped) -- :), :p, :(, o:, x: -- or
eastern ^_^.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | text | string |             Input text. |


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | re.sub | string |             The text with the emoticons replaced by repl. |



# Spacy

``` python
 class Spacy(object)
```

Wrapper to spaCy.io. From their docs @ http://http://spacy.io/docs/.

"spaCy consists of a vocabulary table that stores lexical types, a
pipeline that produce annotations, and three classes to manipulate
document, span and token data. The annotations are predicted using
statistical models, according to specifications that follow common
practice in the research community."
spaCy is currently used in Omesa to provide the English part of the
backbone. It's faster than CoreNLP, and Python <3. While spaCy can also
extract things such as NER (it lacks sentiment and co-reference), this
is currently not enabled for Omesa.

---------

## Methods



| Function    | Doc             |
|:-------|:----------------|
        | parse | Extract spaCy tags. |



### parse

``` python
    parse(text)
```


Extract spaCy tags.

Convert raw text instance into spaCy format. Currently only returns
token, lemma, POS.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | text | string | A raw string of characters. |


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | instance | list | The token, lemma, POS list that can be used in featurizers. |



# Frog

``` python
 class Frog(object)
```

Wrapper to python-frog, loaded from LaMachine.

Excerpt from the documentation @ http://ilk.uvt.nl/frog/:
Frog is an integration of memory-based natural language processing (NLP)
modules developed for Dutch. All NLP modules are based on Timbl, the
Tilburg memory-based learning software package. Recently, a dependency
parser, a base phrase chunker, and a named-entity recognizer module were
added. Where possible, Frog makes use of multi-processor support to run
subtasks in parallel.
Frog is currently used in Omesa to provide the Dutch part of the backbone.
As the other backbones, it currently only uses a subset of features. Full
list of potential extractions (not enabled for Omesa) are:
- Morphological segmentation (according to MBMA).
- Confidence in the POS tag, a number between 0 and 1, representing the
probability mass assigned to the best guess tag in the tag distribution.
- Named entity type, identifying person (PER), organization (ORG), location
(LOC), product (PRO), event (EVE), and miscellaneous (MISC), using a BIO
(or IOB2) encoding.
- Base (non-embedded) phrase chunk in BIO encoding.
- Token number of head word in dependency graph (according to CSI-DP).
- Type of dependency relation with head word.

---------

## Methods



| Function    | Doc             |
|:-------|:----------------|
        | parse | Extract frog tags. |



### parse

``` python
    parse(text)
```


Extract frog tags.

Convert raw text instance into Frog format. Currently only returns
token, lemma, POS.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | text | string | A raw string of characters. |


| Returns    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | instance | list | The token, lemma, POS list that can be used in featurizers. |
        
