# 🐣 Innovation Sweet Spots: Digital technologies for the early years

**_Analysis of trends in research funding, publication, patent and venture funding around digital technologies to support the development of young children._**

## 👋 Welcome!

In the UK, too many young children don't get what they need to develop and thrive in the early years. Through our fairer start mission, Nesta is seeking to tackle this. In this work, we considered the trends in AI and other digital technologies aiming to support the early-years sector and parents to better detect and manage child development needs. [Read more about the project and its findings.](https://www.nesta.org.uk/data-visualisation-and-interactive/innovation-sweet-spots-digital-technologies-early-years//)

In this repo, we keep the code for collecting and analysing our innovation datasets.

_NB: This codebase is at a prototyping stage, where some of the code is still living in Jupyter notebooks whereas some utilities have already been neatly factored out into modules. Do [contact us](mailto:karlis.kanders@nesta.org.uk) if you'd like to reuse parts of this, and we'll be happy to help. We will refactor the codebase and transfer reusable parts into our automated horizon scanning repo in the near future_

## Setup

- Create a conda environment and install pip
- Install the requirements from both `requirements_dev.txt` (these are common to all projects that use this cookiecutter) and `requirements.txt` (specific to this project)
- Run `pip install -e .` to install all functions from `discovery_child_development`
- Create a `.env` file at the root of the project and populate it following the template in `env_template.txt`

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
