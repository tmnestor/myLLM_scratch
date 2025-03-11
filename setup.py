from setuptools import setup, find_packages

setup(
    name="sentence_transformer",
    version="1.0.0",
    description="A clean implementation of transformer models for sentence embeddings",
    author="T.M. Nestor",
    author_email="tod.m.nestor@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "transformers>=4.6.0",
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "safetensors>=0.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
