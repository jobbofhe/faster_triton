from setuptools import setup, find_packages

setup(
    name="triton-llama3-training",
    version="0.1.0",
    description="Triton-based Llama3-8B Training and Inference Framework",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "triton>=2.1.0",
        "transformers>=4.36.0",
        "sentencepiece>=0.1.99",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "datasets>=2.15.0",
        "accelerate>=0.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "triton-llama3-train=examples.train_llama3:main",
            "triton-llama3-infer=examples.infer_llama3:main",
        ],
    },
)
