"""
Execfile is a tool that enables open a python script, extracting the
file-level docstring, executing the file, and saving the resulting
matplotlib figures.
"""
import sys
import os
import traceback
import token
import tokenize
import gc

import matplotlib
matplotlib.use('Agg') #don't display plots
import pylab as plt

class ExecFile:
    """Execute the file and store the output, docstring, and
    sequence of matplotlib figures
    """
    def __init__(self, filename, execute=True, print_output=True):
        self.filename = filename
        self.extract_docstring()
        self.figlist = []
        self.output = ''
        self.stdout = sys.stdout
        self.print_output = print_output
        if execute:
            self.execute_file()

    def save_figures(self, fmt):
        from matplotlib import colors
        for fig in self.figlist:
            figfile = fmt % fig.number
            print("saving", figfile)

            # if black background, save with black background as well.
            if colors.colorConverter.to_rgb(fig.get_facecolor()) == (0, 0, 0):
                fig.savefig(figfile,
                            facecolor='k',
                            edgecolor='none')
                fig.close()
            else:
                fig.savefig(figfile)
                fig.close()

    def write(self, s):
        if self.print_output:
            self.stdout.write(s)
        self.output += s

    def flush(self):
        if self.print_output:
            self.stdout.flush()

    def extract_docstring(self):
        """ Extract a module-level docstring
        """
        lines = open(self.filename).readlines()
        start_row = 0
        if lines[0].startswith('#!'):
            lines.pop(0)
            start_row = 1

        docstring = ''
        first_par = ''
        tokens = tokenize.generate_tokens(lines.__iter__().next)
        for tok_type, tok_content, _, (erow, _), _ in tokens:
            tok_type = token.tok_name[tok_type]
            if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
                continue
            elif tok_type == 'STRING':
                docstring = eval(tok_content)
                # If the docstring is formatted with several paragraphs, extract
                # the first one:
                paragraphs = '\n'.join(line.rstrip()
                                       for line in docstring.split('\n')).split('\n\n')
                if len(paragraphs) > 0:
                    first_par = paragraphs[0]
            break

        self.docstring = docstring
        self.short_desc = first_par
        self.end_line = erow + 1 + start_row


    def execute_file(self):
        """Execute the file, catching standard output
        and matplotlib figures
        """
        dirname, fname = os.path.split(self.filename)
        print('plotting %s' % fname)

        # close any currently open figures
        plt.close('all')

        # change to file directory for execution
        cwd = os.getcwd()

        try:
            if dirname:
                os.chdir(dirname)

            # set stdout to self in order to catch output (with write method)
            sys.stdout = self

            # execute the file
            with open(os.path.basename(self.filename)) as f:
                code = compile(f.read(), "somefile.py", 'exec')
                exec(code, {'pl' : plt, 'plt' : plt, 'pylab' : plt})

            fig_mgr_list = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
            self.figlist = [manager.canvas.figure for manager in fig_mgr_list]

            self.figlist = sorted(self.figlist,
                                  key = lambda fig: fig.number)

        except:
            print(80 * '_')
            print('{} is not compiling:'.format(fname))
            traceback.print_exc()
            print(80 * '_')
        finally:
            # change back to original directory, and reset sys.stdout
            sys.stdout = self.stdout
            os.chdir(cwd)
            ncol = gc.collect()
            if self.print_output and (ncol > 0):
                print("\n > collected {} unreachable objects".format(ncol))
