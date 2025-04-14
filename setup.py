import setuptools

with open("README.md", "r", encoding='utf-8') as readme_file:
    long_description = readme_file.read()

with open('requirements.txt', 'r', encoding='utf-8') as req_file:
    requirements = [r.strip() for r in req_file.readlines()]

setuptools.setup(
    name="cayleypy",
    version="1.0.0",
    author="CayleyPy Foundation",
    description="Cayley graphs processing using GPU and pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/iKolt/cayleypy",
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    install_requires=requirements,
)