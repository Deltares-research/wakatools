# Wakatools

This is a work in progress.


## Installation (user)
In a Python >= 3.12 environment, install the latest (experimental) version of the main branch directly from GitHub using:

    pip install git+https://github.com/Deltares-research/wakatools.git

## Installation (developer)
Wakatools uses [Pixi](https://github.com/prefix-dev/pixi) for package management and workflows.

With pixi installed, navigate to the folder of the cloned repository and run the following to install all dependencies and the package itself in editable mode:

    pixi install

See the [Pixi documentation](https://pixi.sh/latest/) for more information. Next open
the Pixi shell by running:

    pixi shell

Finally install the pre-commit hooks that enable automatic checks upon committing changes:

    pre-commit install
