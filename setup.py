import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

DEFAULT_DEPENDENCIES = ["setuptools", "tensorflow", "scikit-learn",
                        "scipy", "matplotlib", ]
DEV_DEPENDENCIES = DEFAULT_DEPENDENCIES

setuptools.setup(
    name='attoDNN',
    version='0.0.1',
    author="Tomasz Szoldra",
    author_email="t.szoldra@gmail.com",
    description="attoDNN: toolkit for regression on femtosecond pulse data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",  # TODO URL to documentation
    packages=setuptools.find_packages(),
    install_requires=DEFAULT_DEPENDENCIES,
    extras_require={
        "dev": DEV_DEPENDENCIES,
        # TODO add cuda dependencies
    },
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
)