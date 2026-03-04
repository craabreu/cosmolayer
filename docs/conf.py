# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import os
import sys

import cosmolayer
from cosmolayer import cosmosac

sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------
# Auto-generate API rst files
# ---------------------------------------------------------------------


def create_class_rst_file(cls, module_name="cosmolayer"):
    name = cls.__name__
    excluded = ["yaml_tag"]
    attributes = []
    methods = []
    for member_name, member_value in cls.__dict__.items():
        if member_name.startswith("_") or member_name in excluded:
            continue
        if isinstance(member_value, property):
            attributes.append(member_name)
        elif isinstance(member_value, (classmethod, staticmethod)) or callable(member_value):
            methods.append(member_name)
        else:
            attributes.append(member_name)
    attributes = sorted(attributes)
    methods = sorted(methods)

    included_attributes = [
        (
            f"    .. autoattribute:: {attr}\n"
            "        :no-index-entry:\n"
        )
        for attr in attributes
    ]
    included_methods = [
        (
            f"    .. automethod:: {method}\n"
            "        :no-index-entry:\n"
        )
        for method in methods
    ]

    with open(f"api/{name}.rst", "w") as f:
        f.writelines(
            [
                f"{name}\n",
                "=" * len(name) + "\n\n",
                f".. currentmodule:: {module_name}\n",
                f".. autoclass:: {name}\n",
                "    :member-order: alphabetical\n\n",
            ]
            + ["    .. rubric:: Attributes\n\n"] * bool(included_attributes)
            + included_attributes
            + ["    .. rubric:: Methods\n\n"] * bool(included_methods)
            + included_methods
        )


def create_function_rst_file(func, module_name="cosmolayer"):
    name = func.__name__
    with open(f"api/{name}.rst", "w") as f:
        f.writelines(
            [
                f"{name}\n",
                "=" * len(name) + "\n\n",
                f".. currentmodule:: {module_name}\n",
                f".. autofunction:: {name}\n",
            ]
        )


def create_constant_rst_file(const_name, const_value, module_name="cosmolayer"):
    with open(f"api/{const_name}.rst", "w") as f:
        f.writelines(
            [
                f"{const_name}\n",
                "=" * len(const_name) + "\n\n",
                f".. currentmodule:: {module_name}\n",
                f".. py:data:: {const_name}\n\n",
                f"**Type:** {type(const_value).__name__}\n\n",
                f"**Value:** ``{const_value!r}``\n",
            ]
        )


def select(func, module):
    return [item for item in module.__dict__.values() if func(item)]


def create_module_docs(module, module_name, title, output_dir="api"):
    classes = select(inspect.isclass, module)
    functions = select(inspect.isfunction, module)
    submodules = select(inspect.ismodule, module)

    constants = [
        (name, value)
        for name, value in module.__dict__.items()
        if not (name.startswith("_") or value in (classes + functions + submodules))
    ]

    if not classes and not functions and not constants:
        return None

    if module_name == "cosmolayer":
        toctree = "core.rst"
    else:
        module_short_name = module_name.split(".")[-1]
        toctree = f"{module_short_name}.rst"

    with open(f"{output_dir}/{toctree}", "w") as f:
        f.write(f"{title}\n" f"{'=' * len(title)}\n\n")

        if classes:
            f.write("Classes\n-------\n\n.. toctree::\n    :titlesonly:\n\n")
            for item in sorted(classes, key=lambda x: x.__name__):
                f.write(f"    {item.__name__}\n")
                create_class_rst_file(item, module_name)
            f.write("\n")

        if functions:
            f.write("Functions\n---------\n\n.. toctree::\n    :titlesonly:\n\n")
            for item in sorted(functions, key=lambda x: x.__name__):
                f.write(f"    {item.__name__}\n")
                create_function_rst_file(item, module_name)
            f.write("\n")

        if constants:
            f.write("Constants\n---------\n\n.. toctree::\n    :titlesonly:\n\n")
            for const_name, const_value in sorted(constants, key=lambda x: x[0]):
                f.write(f"    {const_name}\n")
                create_constant_rst_file(const_name, const_value, module_name)
            f.write("\n")

        f.write(".. testsetup::\n\n    from cosmolayer import *")

    return toctree


main_toctree = create_module_docs(cosmolayer, "cosmolayer", "Core API")
sac_toctree = create_module_docs(cosmosac, "cosmolayer.cosmosac", "COSMO-SAC")

with open("api/index.rst", "w") as f:
    entries = []
    if main_toctree:
        entries.append(f"    {main_toctree}\n")
    if sac_toctree:
        entries.append(f"    {sac_toctree}\n")

    f.write(
        "API Reference\n"
        "=============\n\n"
        ".. toctree::\n"
        "    :maxdepth: 2\n"
        "    :titlesonly:\n\n"
        + "".join(entries)
    )

# ---------------------------------------------------------------------
# Project info
# ---------------------------------------------------------------------

version = os.getenv("COSMOLAYER_VERSION", cosmolayer.__version__)
project = f"CosmoLayer {version}"
copyright = r"2026 C. Abreu"
author = "Charlles Abreu"
release = ""

# ---------------------------------------------------------------------
# General config
# ---------------------------------------------------------------------

needs_sphinx = "4.4"

extensions = [
    "sphinxarg.ext",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
    # modern visuals
    "sphinx_design",
    "sphinxext.opengraph",
]

autosummary_generate = False

napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = "index"

language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Better modern code highlighting
pygments_style = "friendly"
pygments_dark_style = "native"

toc_object_entries_show_parents = "hide"
add_function_parentheses = False

# ---------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "logo": {"text": project},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/craabreu/cosmolayer",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": False,   # important for generated .rst pages
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "search-field"],
    "secondary_sidebar_items": {
        "**": ["page-toc"],  # keep it simple and clean
    },
}

html_sidebars = {
    "api/**": ["sidebar-nav-bs"],
    "getting_started": [],
    "visualization": [],
    "references": [],
}


html_context = {
    "github_user": "craabreu",
    "github_repo": "cosmolayer",
    "github_version": "main",
    "doc_path": "docs",
}

# OpenGraph (nice link previews)
ogp_site_name = project
ogp_description_length = 200

def setup(app):
    app.add_css_file("css/custom.css")

# ---------------------------------------------------------------------
# HTML Help / LaTeX / man / Texinfo
# ---------------------------------------------------------------------

htmlhelp_basename = "cosmolayerdoc"

latex_documents = [
    (master_doc, "cosmolayer.tex", "CosmoLayer Documentation", "cosmolayer", "manual"),
]

man_pages = [(master_doc, "cosmolayer", "CosmoLayer Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "cosmolayer",
        "CosmoLayer Documentation",
        author,
        "cosmolayer",
        "Differentiable COSMO-Type Activity Coefficient Layer",
        "Miscellaneous",
    ),
]

# ---------------------------------------------------------------------
# Extension config
# ---------------------------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_typehints_format = "short"

bibtex_bibfiles = ["refs.bib"]

extlinks = {
    "PyTorch": ("https://pytorch.org/docs/stable/%s.html", "pytorch.%s"),
}

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

plot_include_source = True
