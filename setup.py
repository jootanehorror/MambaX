from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mambax",
    version="0.4.0",  
    author="Oleg Kufa",
    author_email="os.schischkin@gmail.com",
    description="Optimized Mamba implementation with chunk processing and ONNX export",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",  
        "numpy>=1.21.0",
    ],
    python_requires=">=3.10",  
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="mamba, pytorch, deep learning",
    project_urls={
        "Source Code": "https://github.com/yourusername/mambax",
        "Bug Tracker": "https://github.com/yourusername/mambax/issues",
    },
)
