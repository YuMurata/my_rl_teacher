from setuptools import setup, find_packages

setup(
    name='my_rl_teacher',
    version='0.0.1',
    description='predict evaluate function from human preference',
    author='Yu Murata',
    author_email='me@kennethreitz.com',
    url='https://github.com/YuMurata/my_rl_teacher',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
        'tensorflow',
        'my_deep_learning'
        ],
    dependency_links=['git + https://github.com/YuMurata/my_deep_learning.git#egg=my_deep_larning-0.0.1']
)