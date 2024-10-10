from setuptools import setup, find_packages

setup(
    name='bft_qr_reader',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'opencv-contrib-python',
        'zxing-cpp',
        'fastapi',
        'uvicorn',
        'python-magic',
        'pillow',
        'python-multipart',
        'qrdet @ git+https://github.com/wmetcalf/qrdet.git',
        'QReader @ git+https://github.com/wmetcalf/QReader.git',        
    ],
    package_data={
        'bft_qr_reader': ['models/*']
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'bft_qr_reader=bft_qr_reader.bft_qr_reader:main',
        ],
    },
    author='Will Metcalf',
    author_email='william.metcalf@gmail.com',
    description='A QR code reader implementation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wmetcalf/bft_qr_reader',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
