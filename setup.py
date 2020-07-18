import setuptools

setuptools.setup (name = 'splintr',
        version = '0.0.5',
        description = 'One-Line Single-Machine Multi-GPU DNN Training',
        url = 'https://github.com/splintr-ml/splintr',
        author = 'splintr-ml',
        author_email = 'splintrml@gmail.com',
        install_requires = ['torch', 'numpy'],
        packages = setuptools.find_packages(),
        classifiers = ['Development Status :: 3 - Alpha', 'Environment :: GPU', 'Programming Language :: Python :: 3'])
