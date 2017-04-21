from setuptools import find_packages, setup

install_requires = ['pkgconfig', 'cython', 'numpy', 'six', 'unittest2',
                    'fuel>=0.2.0', 'h5py>=2.6.0', 
                    'keras>=1.1.2.1', 'theano>=0.8.2a0',
                    'n3lu_cacdi>=0.2.4', 'fuel-cacdi']
setup(
    name='cacdi_attention_model',
    version='0.0.1',
    description='Hierarchical attention network for CACDI',
    license='secret',
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    dependency_links=[
        "git+ssh://git@git.labs.nuance.com/dhdaines/n3lu_cacdi.git@0.2.4#egg=n3lu_cacdi-0.2.4",
        "git+ssh://git@git.labs.nuance.com/xavier.bouthillier/fuel-cacdi.git#egg=fuel-cacdi-0.2.0"
    ],
    entry_points={
    'console_scripts': [
        'check_logs=results.check_logs:main',
        'get_results=results.get_results:main',
        'attention_imdb_exp_with_fuel=attention_imdb_exp_with_fuel:parse_commands'
        ]
    }

)
