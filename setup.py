from setuptools import setup, find_packages
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="triple-riding-detection",
    version="0.1.0",
    author="Anshika",
    author_email="your.email@example.com",  # Update this with your email
    description="YOLOv8 based triple riding detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anshika-ux/Coco_model_triple_riding_detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'datasets': ['**/*.yaml', '**/*.txt', '**/*.jpg', '**/*.png'],
    },
)
