from setuptools import setup, find_packages

setup(
    name='evtpooling',
    version='0.1.2',
    author='J.T. Kim',
    description='evtpooling contains the framework needed to improve tail risk forecasts',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'pandas>=2.3.0',
        'numpy>=2.2.0',
        'scikit-learn>=1.7.0',
        'fuzzywuzzy>=0.18.0',
        'python-Levenshtein>=0.27.0',
    ],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ]
)

# TO IMPROVE CODE: run flake8 src tests

# TO UPLOAD TO TEST PYPI: python -m twine upload --repository testpypi dist/*
# TO UPLOAD TO PYPI: python -m twine upload dist/*
