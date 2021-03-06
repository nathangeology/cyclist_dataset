from setuptools import setup

setup(name='data_science_layer',
      version='0.1',
      description='Wrapper for various data science packages to make pipelines',
      url='coming soon',
      author='Nathaniel Jones',
      author_email='nathan.geology@gmail.com',
      license='MIT',
      packages=['data_science_layer'],
      install_requires=[  # 'cx_Oracle',
          'pandas',
          'numpy',
          # 'pyodbc',
          'scipy',
          # 'patsy',
          'sqlalchemy',
          'joblib',
          'scikit-learn',
          # 'torch',
          # 'xgboost',
          # 'progress',
          'scikit-image',
          'imblearn',
          # 'geopandas',
          # 'altair',
          'numba',
          'psutil',
          'lasio',
          'matplotlib',
          'pyepsg',
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )
