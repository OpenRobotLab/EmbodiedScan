from setuptools import find_packages, setup

if __name__ == '__main__':
    my_packages = find_packages()
    print(my_packages)
    setup(
        name='mmscan',
        version='0.0.0',
        author='linjingli',
        author_email='rbler1234@sjtu.edu.cn',
        description='MMScan tools for data loading and evaluation',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/yourusername/your_library',
        packages=my_packages,
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.6',
        install_requires=[],
    )
