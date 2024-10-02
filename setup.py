from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sae_network_analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Owen Parsons',
    description='A package for using network analysis to explore SAE features',
    license='MIT',
    url='https://github.com/owenparsons/sae_network_analysis',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
