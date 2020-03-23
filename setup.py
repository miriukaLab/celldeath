import setuptools



with open('README.md', 'r') as fh:
     long_description = fh.read()




setuptools.setup(
     name='celldeath',
     version='0.9.1',
     author='Santiago Miriuka, Alejandro La Greca, Nelba Pérez',
     author_email='smiriuka@fleni.org.ar, ale.lagreca@gmail.com, nelbap@hotmail.com',
     description='A tool to identify cell death based on deep learning',
     long_description=long_description, 
     long_description_content_type='text/markdown', 
     url='https://github.com/sgmiriuka/celldeath', 
     packages=setuptools.find_packages(),
     install_requires=['fastai>=1.4',
                         'image-slicer==0.3.0',
                         'matplotlib>=3.1.1'], 
     classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Medical Science Apps.'
     ],
     python_requires='>=3.6'
)

