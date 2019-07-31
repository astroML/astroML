import os
from example_builder import ExampleBuilder


RST_TEMPLATE = """

.. _%(sphinx_tag)s:

%(docstring)s

%(image_list)s

.. raw:: html

    <div class="toggle_trigger"><a href="#">

**Code output:**

.. raw:: html

    </a></div>
    <div class="toggle_container">

.. literalinclude:: %(stdout)s

.. raw:: html

    </div>
    <div class="toggle_trigger" id="start_open"><a href="#">

**Python source code:**

.. raw:: html

    </a></div>
    <div class="toggle_container">

.. literalinclude:: %(fname)s
    :lines: %(end_line)s-

.. raw:: html

    </div>
    <div align="right">

:download:`[download source: %(fname)s] <%(fname)s>`

.. raw:: html

    </div>

"""


def main(app):
    target_dir = os.path.join(app.builder.srcdir, 'book_figures')
    book_figures_path = os.environ.get('BOOK_FIGURES_PATH', '../book_figures')
    source_dir = os.path.abspath(book_figures_path)

    try:
        plot_gallery = eval(app.builder.config.plot_gallery)
    except TypeError:
        plot_gallery = bool(app.builder.config.plot_gallery)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    EB = ExampleBuilder(source_dir, target_dir,
                        execute_files=plot_gallery,
                        contents_file='contents.txt',
                        dir_info_file='README.rst',
                        dir_footer_file='FOOTER.rst',
                        sphinx_tag_base='book_fig',
                        template_example=RST_TEMPLATE)
    EB.run()


def setup(app):
    app.connect('builder-inited', main)
    #app.add_config_value('plot_gallery', True, 'html')
