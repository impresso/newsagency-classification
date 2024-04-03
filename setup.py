from setuptools import setup, find_packages

setup(
    name='newsagency-classification',
    version='1.0.0',
    author='Lea Marxen',
    author_email='your@email.com',
    description='News Agency Recognition',
    long_description='News Agency Recognition',
    long_description_content_type='text/markdown',
    url='https://github.com/impresso/newsagency-classification',
    packages=find_packages(),
    install_requires=[
        'dependency1',
        'dependency2>=1.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
