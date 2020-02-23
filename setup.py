from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

os.environ["CC"] = "/usr/bin/gcc-7"
os.environ["CXX"] = "/usr/bin/g++-7"
setup(name='game_cpp',
      ext_modules=[cpp_extension.CppExtension('game_cpp', ['game.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})