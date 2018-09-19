from setuptools import setup

#REFERENCE: [7]

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='brightcomets',
    url='https://github.com/joemasiero/BrightComets',
    author='Nathan Blair',
    author_email='ncblair@me.com',
    # Needed to actually package something
    packages=['brightcomets'],
    # Specify Python version
    python_requires='~=3.6.5',
    # Needed for dependencies
    install_requires=['numpy', 'arrr', 'Pillow', 'astropy', 'matplotlib', \
                    'pyds9', 'pyregion', 'scikit_image', 'scipy', 'tensorflow'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='NASA',
    description='An example of a python package from pre-existing code',
    long_description=open('README.rst').read(),
)