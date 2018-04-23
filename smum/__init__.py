import os

for dir_name in ['data', 'FIGURES', 'temp']:
    directory = os.path.join(os.getcwd(), dir_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
