import os
from example_builder import ExampleBuilder

def main(app):
    target_dir = os.path.join(app.builder.srcdir, 'auto_examples')
    source_dir = os.path.abspath(app.builder.srcdir +  '/../' + 'examples')

    try:
        plot_gallery = eval(app.builder.config.plot_gallery)
    except TypeError:
        plot_gallery = bool(app.builder.config.plot_gallery)

    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    EB = ExampleBuilder(source_dir, target_dir,
                        execute_files=plot_gallery,
                        dir_info_file='README.rst',
                        sphinx_tag_base='example')
    EB.run()
                            

def setup(app):
    app.connect('builder-inited', main)
    app.add_config_value('plot_gallery', True, 'html')
