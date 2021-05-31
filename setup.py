from setuptools import setup, find_packages


setup(name='vi_dpp',
      version='0.1.0',
      description='Variational inference for decomposable point processes',
      author='Aristeidis Panos',
      author_email='ares.panos@warwick.ac.uk',
      packages=find_packages('.'),
      zip_safe=False)
