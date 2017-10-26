mkdir ./examples/FIGURES_rst
mkdir examples_rst
jupyter-nbconvert --config ipython_nbconvert_config.py
mv ./examples/*.rst ./examples_rst
mv ./examples/FIGURES_rst ./examples_rst/FIGURES_rst
