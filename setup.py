from pathlib import Path

import setuptools

import mdmm


setuptools.setup(
    name='mdmm',
    version=mdmm.__version__,
    description='The Modified Differential Multiplier Method (MDMM) for PyTorch.',
    long_description=(Path(__file__).resolve().parent / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/crowsonkb/mdmm',
    author='Katherine Crowson',
    author_email='crowsonkb@gmail.com',
    license='MIT',
    license_files=['LICENSE'],
    packages=['mdmm'],
    install_requires=['dataclasses>=0.8;python_version<"3.7"',
                      'torch>=1.7.1'],
    python_requires=">=3.6",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
