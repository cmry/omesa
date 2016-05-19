"""Converts sklearn-styled docstrings to markdown."""

from sys import argv

# Author:       Chris Emmery
# License:      MIT

ERR = '''
- """ used for non-docstring strings.
- Class or function Documentation has newline after opening """.
'''


class Doc2Markdown(object):
    """Documentation to Markdown."""

    def __init__(self, file_in, file_out):
        """Handle in/out to readers and parsers."""
        self.file_in = file_in
        self.file_out = file_out
        self.markdown = ''

        self.md_table = '''

| {0}    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        '''
        self.md_table_row = '''| {0} | {1} | {2} |
        '''

        self.md_table_alt = '''

| {0}    | Doc             |
|:-------|:----------------|
        '''
        self.md_table_row_alt = '''| {0} | {1} |
        '''

        self.md_code = '``` python \n {0} \n```\n\n'

        self.read()

    def read(self):
        """Open pytyhon code file, spit out markdown file."""
        with open(self.file_in, 'r') as file_in:
            file_open = file_in.read()
            classes = file_open.split('\nclass ')
            module_doc = classes.pop(0)  # top of the file docstring
            self.markdown = [self.md_title(module_doc)]
            self.handle_classes(classes)
        with open(self.file_out, 'w') as file_out:
            file_out.write('\n'.join(self.markdown))

    def handle_classes(self, classes):
        """Given class docs only, write out their docs and functions."""
        for clss in classes:
            clss = " class " + clss  # got cut-off in split
            clss_str = clss.split('"""\n')[:-1]
            # process all the class information
            self.markdown.append(
                self.md_class_doc(self.split_doc(clss_str.pop(0))))
            # replace object part with __init__ params
            self.markdown[1] = \
                self.md_class_func(self.markdown[1], clss_str)
            # start handling def
            self.markdown.append(
                self.md_funcs_doc([self.split_doc(func) for func in clss_str]))

    @staticmethod
    def split_doc(doc):
        """Split a python file on docstring."""
        lines = [x for x in doc.split('\n') if x]
        name, det = '', False
        # for i, l in enumerate(lines):
        #     print("{0}: ".format(i+1), l)
        # skip lines until class/def, end if :
        while ':' not in name:
            try:
                if 'class ' in lines[0] or 'def ' in lines[0]:
                    det = True
            except IndexError:
                exit("Malformed code. \n\nSome common errors: {0}".format(ERR))
            line = lines.pop(0)
            if det:
                name += line

        parts = {'name': name}
        buffer_to = 'title'
        # if ----- is found, set title to above, else buffer to name
        for i in range(0, len(lines)):
            line = lines[i]
            if '---' in line:
                buffer_to = lines[i-1]
                continue
            if not parts.get(buffer_to):
                parts[buffer_to] = [buffer_to]
            if parts.get(buffer_to) and line:
                parts[buffer_to].append(line)
        # if line is shorter than 3, likely that there is no previous section
        return {k.replace('  ', ''):
                (v[:-1] if len(v) > 2 else v)[1:] for k, v in parts.items()}

    @staticmethod
    def md_title(title_part, class_line=False):
        """Handle title and text parts for main and class-specific."""
        buff, start, second = [], False, False
        # pretty convoluted function just to append # to first line and clean
        for line in title_part.split('\n'):
            line = line.replace('  ', '')
            if line.startswith('"""') and not start:
                title = '# ' + line.replace('"""', '')[:-1]
                if class_line:
                    title = title[2:] + '.'
                buff.append(title)
                start = True
            elif line.startswith('"""') and start:
                start = False
                return '\n'.join(buff)
            elif start:
                if not second and not line.startswith('\n'):
                    line = "\n" + line
                    second = True
                buff.append(line)
        return '\n'.join(buff)

    def md_par_att(self, doc, name):
        """Place parameters and attributes in a table overview."""
        head, buff, lines = '', (), []
        table = self.md_table.format(name)
        # given var : type \n description, splits these up into 3 cells
        if doc:
            for row in doc:
                if ' : ' in row and not buff:
                    head = row.split(' : ')
                    for part in head:
                        buff += (part.replace('  ', ''), )
                elif ' : ' in row and buff:
                    buff += (''.join(lines), )
                    table += self.md_table_row.format(*buff)
                    buff, lines = (), []
                elif buff:
                    lines.append(row.replace('\n', ' '))
            if buff:
                buff += (''.join(lines).replace('  ', ''), )
                table += self.md_table_row.format(*buff)
            return table
        else:
            return ''

    def md_examples(self, doc):
        """Section text and code in the example part of the doc."""
        head = '\n\n------- \n\n##Examples\n\n{0}'
        order = []
        text, code = '', ''
        # if we find some code related beginnings (>>>, ...) buffer to code,
        # else we regard it as text. Record order so that multiple blocks of
        # text and code are possible.
        if doc:
            for row in doc:
                if '>>>' in row or '...' in row:
                    if text:
                        order.append(text)
                        text = ''
                    row = row.replace('    >>>', '>>>')
                    row = row.replace('    ...', '...')
                    code += row + '\n'
                else:
                    if code:
                        order.append(self.md_code.format('\n' + code))
                        code = ''
                    row = row.replace('  ', '')
                    text += row + '\n'
            if text or code:
                order.append(text)
                order.append(self.md_code.format('\n' + code))
            return head.format('\n'.join(order).replace('.\n', '.\n\n'))
        else:
            return ''

    def md_class_func(self, doc, clss):
        """Replace class ...(object) with __init__ arguments."""
        init_doc = self.split_doc(clss.pop(0))['name']
        init_doc = init_doc.replace('  ', '')
        init_doc = init_doc.replace(' def __init__(self, ', '(')
        return doc.replace('(object)', init_doc)

    def md_funcs_doc(self, func_doc):
        """Merge all function elements in one string."""
        mark_doc, mark_head = '', '\n--------- \n\n## Methods \n\n {0} \n {1}'
        func_table = self.md_table_alt.format('Function')
        for func in func_doc:
            name = func['name'].replace('(', ' (').split()[1]
            title = self.md_title('\n'.join(func['title']), class_line=True)
            func_table += self.md_table_row_alt.format(name,
                                                       title.split('\n')[0])
            mark_doc += '\n\n### ' + name + '\n\n'

            cod = self.md_code.format(func['name'].replace('def ', ''))
            cod = cod.replace('self', '').replace('(, ', '(')

            mark_doc += cod
            mark_doc += '\n' + title
            mark_doc += self.md_par_att(func.get('Parameters'), 'Parameters')
            mark_doc += self.md_par_att(func.get('Returns'), 'Returns')
        return mark_head.format(func_table, mark_doc)

    def md_class_doc(self, class_doc):
        """Merge all class section elements in one string."""
        mark_doc = ''
        mark_doc += '\n\n# {0} \n\n'.format(
            class_doc['name'].replace('(', ' (').split()[1])
        mark_doc += self.md_code.format(class_doc['name'])
        mark_doc += self.md_title('\n'.join(class_doc['title']),
                                  class_line=True)
        mark_doc += self.md_par_att(class_doc.get('Parameters'), 'Parameters')
        mark_doc += self.md_par_att(class_doc.get('Attributes'), 'Attributes')
        mark_doc += self.md_examples(class_doc.get('Examples'))
        return mark_doc

if __name__ == '__main__':
    D2M = Doc2Markdown(argv[1], argv[2])
