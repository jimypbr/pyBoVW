from setuptools import setup, find_packages
import sys, os

version = '0.1'

setup(name='bovw',
      version=version,
      description="Bag-of-Visual-Words feature extractor",
      long_description="""\
Tools for extracting Bag-of-Visual-Word feature vectors from images based on the sklearn API""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='',
      author='James Briggs',
      author_email='jimy.pbr@gmail.com',
      url='',
      license='BSD',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
