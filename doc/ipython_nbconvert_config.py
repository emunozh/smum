c = get_config()

#Export all the notebooks in the current directory to the sphinx_howto format.
c.NbConvertApp.notebooks = ['examples/*.ipynb']
c.NbConvertApp.export_format = 'rst'
c.NbConvertApp.output_files_dir = 'FIGURES_rst/'
