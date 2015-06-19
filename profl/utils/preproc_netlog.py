import re
import HTMLParser


def floodings(text):
    '''
    Returns a list of tuples (complete_flooding, flooded_item),
    e.g.('iii', 'i')
    '''
    floodings = re.findall(r"((.)\2{2,})", text)
    floodings = [tup for tup in floodings if tup[0] != '...']
    return floodings


def restore_html_symbols(text):
    """
    Restore HTML symbols
    ====================

    Parameters
    -----
    text : string
        Input text.

    Returns
    -----
    text : string
        The text with the restored HTML symbols.

    Examples:
    -----
        &quot; --> "
        &amp; --> &
        &lt; --> <
        &gt; --> >
    """
    html_parser = HTMLParser.HTMLParser()
    return html_parser.unescape(text)


def replace_netlog_tags(text):
    """
    Replace Netlog tags
    ===================
    Replace all tags with [], which are included in the Netlog data,
    with a tag consisting of capital letters surrounded by underscores.
    Typography tags such as [b], [/b], [u], [/u], [i], [/i] (for bold,
    underlined and italics) are removed.

    The new tags are:
    - _PHOTO_, to replace photo tags such as [photo]116157181[/photo]
    - _VIDEO_, to replace video tags such as [video]nl-9159440[/video]
    - _URL_, to replace url tags such as [url=http://www.adres.be/]Adres[/url] or [/url]
    - _EMOTICON_, to replace remaining tags such as [love], [@hug], [#clap_anim]

    Parameters
    -----
    text : string
        Input text.

    Returns
    -----
    text : string
        The text in which the Netlog tags with [] have been replaced.
    """
    # REGULAR EXPRESSIONS:
    # To extract strings like [url=http://www.adres.be/]Adres[/url]
    url_tag1 = re.compile(r"\[url=[^\]]+\][^\[]+\[\/url\]")
    # To extract [/url] (which is present separately in OLDNETLOG)
    url_tag2 = re.compile(r"\[\/url\]")
    # to extract strings like [photo]116157181[/photo]
    photo_tag = re.compile(r"\[photo\][^\]]+\[\/photo\]")
    # To extract strings like [video]nl-9159440[/video]
    video_tag = re.compile(r"\[video\][^\[\]]+\[\/video\]")
    # To extract typography tags [b], [/b], [u], [/u], [i], [/i]
    # (for bold, underlined, italics)
    typogr_tag = re.compile(r"\[\/?[biu]\]")
    # To extract all other tags in [], which are mainly emoticon tags,
    # like [love], [@hug], [#clap_anim]
    general_tag = re.compile(r"\[[^\[\]]+\]")
    # EDIT TEXT:
    # Replace [] tags with all-caps tags surrounded by underscores
    text = url_tag1.sub('_URL_', text)
    text = url_tag2.sub('_URL_', text)
    text = photo_tag.sub('_PHOTO_', text)
    text = video_tag.sub('_VIDEO_', text)
    # Remove typography tags, as they are not part of the text
    text = typogr_tag.sub('', text)
    # Replace all other tags, which are usually emoticons
    text = general_tag.sub('_EMOTICON_', text)
    return text


def replace_url_email(text):
    """
    Replace URLs and e-mail addresses
    =================================
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
    url1 = re.compile(r"https?\:\/\/\S+")
    url2 = re.compile(r"www\.\S+")
    email = re.compile(r"\S+\@[\w\.\-]+\.\w+")
    text = url1.sub('_URL_', text)
    text = url2.sub('_URL_', text)
    text = email.sub('_EMAIL_', text)
    return text


def replace_emoticons(text, repl_str="_EMOTICON_"):
    """
    Replace emoticons
    =================
    Replace emoticons with a replacement string (default="_EMOTICON_").

    Parameters
    -----
    text : string
        Input text.

    repl_str : string, default "_EMOTICON_"
        String with which emoticons are replaced.

    Returns
    -----
    text : string
        The text with the emoticons replaced by the repl_string.

    Emoticons can be:
    - left-to-right, e.g. :)  O:-D  B^P  XD
    - right-to-left, e.g. (:  (^8  )':  (-:<
    - frontal, e.g. ^_^  (O.O)  (=^.^=)  -_^
    Loosely based on the emoticon list on Wikipedia:
    http://en.wikipedia.org/wiki/List_of_emoticons

    Letters on the outside of the emoticon (as in :P or O-8) are only included
    if they are next to a whitespace character or a punctuation mark (?!.), or
    at the beginning or the end of the string.
    """
    # Parts of left-to-right/right-to-left emoticons
    eyes_lr = r"[\:\;8\=\#\%\*]"
    eyes_lr_let = r"[XxB]"
    mouth_lr = r"[\(\)\[\]\{\}\<\>\\\/\|\$\&\@\#\*\.0]"
    mouth_lr_let = r"[DdPpCcSsOoXxq]"
    mouth_r = r"[\,3]"
    mouth_r_let = r"[LJ]"
    nose = r"[\-\~\^vo]"
    extra_lr = r"[\{\}\<\>\|\=\~03]"
    extra_lr_let = r"[OXx]"
    tear = r"[\']"

    # Parts of frontal emoticons
    eye_fr = r"[0\^\-\>\<\~\.\,\+\*\=\']"
    eye_fr_let = r"[Oo]"
    nose_mouth_fr = r"[\_\-\.0\~oOv]"
    extra_lr_front = r"[\=\<\>]"
    extra_l_front = r"[\(\\]"
    extra_r_front = r"[\)\;\/]"

    # Other emoticons
    cheer = r"\*?\\[oO]\/\*?"
    heart = r"\<[\\\/]3"
    let_in_brack = r"\([a-zA-Z]\)"

    # LEFT-TO-RIGHT EMOTICONS (eyes left, mouth right):

    l_side_ltr1 = extra_lr+"*"+eyes_lr+"+"+tear+"?"
    l_side_ltr2 = extra_lr+"+"+eyes_lr_let+"+"+tear+"?"
    r_side_ltr1 = "(?:"+mouth_lr+"|"+mouth_r+")+"+extra_lr+"*"
    r_side_ltr2 = mouth_lr_let+"+"+extra_lr+"+"

    l_side_ltr_let1 = extra_lr_let+"+"+eyes_lr+"+"+tear+"?"
    l_side_ltr_let2 = extra_lr_let+"*"+eyes_lr_let+"+"+tear+"?"
    r_side_ltr_let1 = "(?:"+mouth_lr+"|"+mouth_r+")+"+extra_lr_let+"+"
    r_side_ltr_let2 = "(?:"+mouth_lr_let+"|"+mouth_r_let+")+"+extra_lr_let+"*"

    emo_ltr1 = l_side_ltr1+nose+"?"+r_side_ltr1
    emo_ltr2 = l_side_ltr1+nose+"?"+r_side_ltr2
    emo_ltr3 = l_side_ltr2+nose+"?"+r_side_ltr1
    emo_ltr4 = l_side_ltr2+nose+"?"+r_side_ltr2

    emo_ltr_let_l1 = l_side_ltr_let1+nose+"?"+r_side_ltr1
    emo_ltr_let_l2 = l_side_ltr_let1+nose+"?"+r_side_ltr2
    emo_ltr_let_l3 = l_side_ltr_let2+nose+"?"+r_side_ltr1
    emo_ltr_let_l4 = l_side_ltr_let2+nose+"?"+r_side_ltr2

    emo_ltr_let_r1 = l_side_ltr1+nose+"?"+r_side_ltr_let1
    emo_ltr_let_r2 = l_side_ltr1+nose+"?"+r_side_ltr_let2
    emo_ltr_let_r3 = l_side_ltr2+nose+"?"+r_side_ltr_let1
    emo_ltr_let_r4 = l_side_ltr2+nose+"?"+r_side_ltr_let2

    emo_ltr_let_lr1 = l_side_ltr_let1+nose+"?"+r_side_ltr_let1
    emo_ltr_let_lr2 = l_side_ltr_let1+nose+"?"+r_side_ltr_let2
    emo_ltr_let_lr3 = l_side_ltr_let2+nose+"?"+r_side_ltr_let1
    emo_ltr_let_lr4 = l_side_ltr_let2+nose+"?"+r_side_ltr_let2

    # RIGHT-TO-LEFT EMOTICONS (eyes right, mouth left):

    l_side_rtl1 = extra_lr+"*"+mouth_lr+"+"
    l_side_rtl2 = extra_lr+"+"+mouth_lr_let+"+"
    r_side_rtl1 = tear+"?"+eyes_lr+"+"+extra_lr+"*"
    r_side_rtl2 = tear+"?"+eyes_lr_let+"+"+extra_lr+"+"

    l_side_rtl_let1 = extra_lr_let+"+"+mouth_lr+"+"
    l_side_rtl_let2 = extra_lr+"*"+mouth_lr_let+"+"
    r_side_rtl_let1 = tear+"?"+eyes_lr+"+"+extra_lr_let+"+"
    r_side_rtl_let2 = tear+"?"+eyes_lr_let+"+"+extra_lr_let+"*"

    emo_rtl1 = l_side_rtl1+nose+"?"+r_side_rtl1
    emo_rtl2 = l_side_rtl1+nose+"?"+r_side_rtl2
    emo_rtl3 = l_side_rtl2+nose+"?"+r_side_rtl1
    emo_rtl4 = l_side_rtl2+nose+"?"+r_side_rtl2

    emo_rtl_let_l1 = l_side_rtl_let1+nose+"?"+r_side_rtl1
    emo_rtl_let_l2 = l_side_rtl_let1+nose+"?"+r_side_rtl2
    emo_rtl_let_l3 = l_side_rtl_let2+nose+"?"+r_side_rtl1
    emo_rtl_let_l4 = l_side_rtl_let2+nose+"?"+r_side_rtl2

    emo_rtl_let_r1 = l_side_rtl1+nose+"?"+r_side_rtl_let1
    emo_rtl_let_r2 = l_side_rtl1+nose+"?"+r_side_rtl_let2
    emo_rtl_let_r3 = l_side_rtl2+nose+"?"+r_side_rtl_let1
    emo_rtl_let_r4 = l_side_rtl2+nose+"?"+r_side_rtl_let2

    emo_rtl_let_lr1 = l_side_rtl_let1+nose+"?"+r_side_rtl_let1
    emo_rtl_let_lr2 = l_side_rtl_let1+nose+"?"+r_side_rtl_let2
    emo_rtl_let_lr3 = l_side_rtl_let2+nose+"?"+r_side_rtl_let1
    emo_rtl_let_lr4 = l_side_rtl_let2+nose+"?"+r_side_rtl_let2

    # FRONTAL EMOTICONS (eye, nose/mouth, eye):

    l_side_fr1 = "(?:"+extra_lr_front+"|"+extra_l_front+")*"+eye_fr
    l_side_fr2 = "(?:"+extra_lr_front+"|"+extra_l_front+")+"+eye_fr_let
    r_side_fr1 = eye_fr+"(?:"+extra_lr_front+"|"+extra_r_front+")*"
    r_side_fr2 = eye_fr_let+"(?:"+extra_lr_front+"|"+extra_r_front+")+"

    emo_fr1 = l_side_fr1+nose_mouth_fr+"?"+r_side_fr1
    emo_fr2 = l_side_fr1+nose_mouth_fr+"?"+r_side_fr2
    emo_fr3 = l_side_fr2+nose_mouth_fr+"?"+r_side_fr1
    emo_fr4 = l_side_fr2+nose_mouth_fr+"?"+r_side_fr2

    emo_fr_let_l1 = eye_fr_let+nose_mouth_fr+"?"+r_side_fr1
    emo_fr_let_l2 = eye_fr_let+nose_mouth_fr+"?"+r_side_fr2

    emo_fr_let_r1 = l_side_fr1+nose_mouth_fr+"?"+eye_fr_let
    emo_fr_let_r2 = l_side_fr2+nose_mouth_fr+"?"+eye_fr_let

    emo_fr_let_lr = eye_fr_let+nose_mouth_fr+"?"+eye_fr_let

    # Specify boundaries required for emoticons with letters on the outside(s)

    bound_l = "(^|\s|[\?\!\.])"
    bound_r = "($|\s|[\?\!\.])"

    # CARRY OUT THE SUBSTITUTIONS

    # Substitute left-to-right emoticons with letters on both sides
    for emo_ltr_let_lr in [emo_ltr_let_lr1, emo_ltr_let_lr2, emo_ltr_let_lr3, emo_ltr_let_lr4]:
        text = re.sub(bound_l+emo_ltr_let_lr+bound_r, "\g<1>"+repl_str+"\g<2>", text)

    # Substitute right-to-left emoticons with letters on both sides
    for emo_rtl_let_lr in [emo_rtl_let_lr1, emo_rtl_let_lr2, emo_rtl_let_lr3, emo_rtl_let_lr4]:
        text = re.sub(bound_l+emo_rtl_let_lr+bound_r, "\g<1>"+repl_str+"\g<2>", text)

    # Substitute left-to-right emoticons with letters on the left side
    for emo_ltr_let_l in [emo_ltr_let_l1, emo_ltr_let_l2, emo_ltr_let_l3, emo_ltr_let_l4]:
        text = re.sub(bound_l+emo_ltr_let_l, "\g<1>"+repl_str, text)

    # Substitute right-to-left emoticons with letters on the left side
    for emo_rtl_let_l in [emo_rtl_let_l1, emo_rtl_let_l2, emo_rtl_let_l3, emo_rtl_let_l4]:
        text = re.sub(bound_l+emo_rtl_let_l, "\g<1>"+repl_str, text)

    # Substitute left-to-right emoticons with letters on the right side
    for emo_ltr_let_r in [emo_ltr_let_r1, emo_ltr_let_r2, emo_ltr_let_r3, emo_ltr_let_r4]:
        text = re.sub(emo_ltr_let_r+bound_r, repl_str+"\g<1>", text)

    # Substitute right-to-left emoticons with letters on the right side
    for emo_rtl_let_r in [emo_rtl_let_r1, emo_rtl_let_r2, emo_rtl_let_r3, emo_rtl_let_r4]:
        text = re.sub(emo_rtl_let_r+bound_r, repl_str+"\g<1>", text)

    # Substitute frontal emoticons without letters on the sides
    for emo_fr in [emo_fr1, emo_fr2, emo_fr3, emo_fr4]:
        text = re.sub(emo_fr, repl_str, text)

    # Substitute frontal emoticons with letters on the left side
    for emo_fr_let_l in [emo_fr_let_l1, emo_fr_let_l2]:
        text = re.sub(bound_l+emo_fr_let_l, "\g<1>"+repl_str, text)

    # Substitute frontal emoticons with letters on the right side
    for emo_fr_let_r in [emo_fr_let_r1, emo_fr_let_r2]:
        text = re.sub(emo_fr_let_r+bound_r, repl_str+"\g<1>", text)

    # Substitute frontal emoticons with letters on both sides
    text = re.sub(bound_l+emo_fr_let_lr+bound_r, "\g<1>"+repl_str+"\g<2>", text)

    # Substitute left-to-right emoticons without letters on the sides
    for emo_ltr in [emo_ltr1, emo_ltr2, emo_ltr3, emo_ltr4]:
        text = re.sub(emo_ltr, repl_str, text)

    # Substitute right-to-left emoticons without letters on the sides
    for emo_rtl in [emo_rtl1, emo_rtl2, emo_rtl3, emo_rtl4]:
        text = re.sub(emo_rtl, repl_str, text)

    # Substitute other emoticons: cheer, single_letter_in_brackets, heart
    text = re.sub(bound_l+cheer+bound_r, "\g<1>"+repl_str+"\g<2>", text)
    text = re.sub(bound_l+let_in_brack+bound_r, "\g<1>"+repl_str+"\g<2>", text)
    text = re.sub(heart, repl_str, text)

    return text
