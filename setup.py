import pybind11
from distutils.core import setup, Extension

ext_modules = [
    Extension(
        'fastnet', # название нашей либы
        ['base/fastnet.cpp', 'base/main.cpp'], # файлики которые компилируем
        include_dirs=[pybind11.get_include()],  # не забываем добавить инклюды pybind11
        language='c++',
    ),
]

setup(
    name='fastnet',
    version='0.0.3',
    author='Gnom',
    author_email='yulia2399@mail.ru',
    description='pybind11 extension',
    ext_modules=ext_modules,
    requires=['pybind11']  # не забываем указать зависимость от pybind11
)