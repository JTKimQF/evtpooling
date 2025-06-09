from setuptools import setup, find_packages

setup(
    name='evtpooling',
    version='0.1.4',
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

# UPLOADING STEPS:
# 1. clean old builds first: Remove-Item dist, build, *.egg-info -Recurse -Force 
# 1a. use this if the above command does not work: Remove-Item dist, build, *.egg-info -Recurse -Force -ErrorAction SilentlyContinue
# 2. build new package: python -m build
# 3. upload to Test PyPI and PyPI:
# TO UPLOAD TO TEST PYPI: python -m twine upload --repository testpypi dist/*
# TO UPLOAD TO PYPI: python -m twine upload dist/*

# TESTING STEPS:
# 1. Create a virtual environment: python -m venv venv-testpypi
# 2. Activate the virtual environment: venv-testpypi\Scripts\Activate
# 3. Install the package from Test PyPI: pip install --index-url https://test.pypi.org/simple/ --no-deps evtpooling
