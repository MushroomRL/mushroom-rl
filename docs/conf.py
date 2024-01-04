# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

#from mushroom_rl import __version__

# General information about the project.
project = u'MushroomRL'
copyright = u'2018-2021 Carlo D\'Eramo, Davide Tateo'
author = u'Carlo D\'Eramo'
# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.youtube']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# To experiment with custom code block styles:
pygments_style = "stata-dark"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True




# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
#version = '.'.join(__version__.split('.')[:-1])

# The full version, including alpha/beta/rc tags.
#release = __version__






root_doc = 'index'
language = "en'"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "piccolo_theme"

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    '/basic_mod.css',
]


html_sidebars = {
   '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
   'using/windows': ['windowssidebar.html', 'searchbox.html'],
}


# The master toctree document.
master_doc = 'index'



html_short_title = 'Mushroom RL'
# Enabling this line will change the nav title from a text to an image:
html_logo = '_static/image3.jpg'
 
html_theme_options = {
   
    
    "source_url": "https://github.com/MushroomRL/mushroom-rl/blob/master/docs/index.rst",
    "dark_mode_code_blocks": True
}





html_show_sourcelink=False



# Output file base name for HTML help builder.
htmlhelp_basename = 'MushroomRLdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    'inputenc': '\\usepackage[utf8x]{inputenc}',
    'preamble': r'''
\DeclareUnicodeCharacter{9989}{\checkmark}
\DeclareUnicodeCharacter{10060}{X}
''',
    'makeindex': '\\usepackage[columns=1]{idxlayout}\\makeindex',
    'printindex': '\\def\\twocolumn[#1]{#1}\\footnotesize\\raggedright\\printindex',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'MushroomRL.tex', u'MushroomRL Documentation',
     u'Carlo D\'Eramo, Davide Tateo', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
main_pages = [
    (master_doc, '/mushroom_rl', u'MushroomRL Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'MushroomRL', u'MushroomRL Documentation',
     author, 'MushroomRL', 'One line description of project.',
     'Miscellaneous'),
]



# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Options for autodoc ---------------------------------------------------

autodoc_member_order = 'bysource'
autodoc_mock_imports = ['torch', 'pybullet', 'pybullet_data', 'pybullet_utils', 'dm_control', 'mujoco', 'glfw',
                        'habitat', 'habitat_baselines', 'habitat_sim', 'igibson',
                        'gym_minigrid']
add_module_names = False












