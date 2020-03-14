import setup



with open('README.md', 'r') as fh:
     long_description = fh.read()




setuptools.setup(
     name='celldeath',
     version='0.9.0',
     author='Santiago Miriuka, Alejandro La Greca, Nelba Pérez',
     author_email='smiriuka@fleni.org.ar, ale.lagreca@gmail.com, nelbap@hotmail.com',
     description='A scprit to identify cell death based on deep learning',
     long_description=long_description, 
     long_description_content_type='text/markdown', 
     url='', 
     packages=setuptools.find_packages(), 
     classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Medical Science Apps.'
     ],
     python_requires='>=3.6'
)

