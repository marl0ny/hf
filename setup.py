from setuptools import Extension, setup
import sys
import os
import platform
import glob
import shutil

print(os.environ['HOME'])

HOME = os.environ['HOME']
PYTHON_VERSION = '.'.join(sys.version.split(' ')[0].split('.')[:2])


INCLUDE_DIRS = [
    f"/usr/include/python{PYTHON_VERSION}", 
    f"{HOME}/.local/lib/python{PYTHON_VERSION}"
    + f"/site-packages/numpy/core/include/numpy",
]
LIB_DIRS = [
    f"/usr/lib/python{PYTHON_VERSION}"
    + f"/config-{PYTHON_VERSION}-x86_64-linux-gnu",
    f"{HOME}/.local/lib/python{PYTHON_VERSION}"
    + "/site-packages/numpy/core/lib/",
]
INCLUDE_DIRS_MACOS = [
    "/opt/homebrew/include", "/usr/include",
    "/Applications/Xcode.app/Contents/Developer/Library/Frameworks"
    + f"/Python3.framework/Versions/{PYTHON_VERSION}/"
    + f"include/python{PYTHON_VERSION}",
    f"{HOME}/Library/Python/3.9/lib/python/site-packages"
    + "/numpy/core/include/numpy"]
LIB_DIRS_MACOS = [
    "/opt/homebrew/lib", "/usr/lib",
    "/Applications/Xcode.app/Contents/Developer/Library/"
    + f"Frameworks/Python3.framework/Versions/{PYTHON_VERSION}/lib",
    "/Applications/Xcode.app/Contents/Developer/Library/Frameworks"
    + f"/Python3.framework/Versions/{PYTHON_VERSION}"
    + f"/lib/python{PYTHON_VERSION}/lib-dynload",
    f"{HOME}/Library/Python/{PYTHON_VERSION}/lib/python"
    + "/site-packages/numpy/core/lib"]

lib_dirs = LIB_DIRS
include_dirs = INCLUDE_DIRS
if platform.system() == 'Windows':
    # TODO!
    pass
elif platform.system() == 'Darwin':
    lib_dirs = LIB_DIRS_MACOS
    include_dirs = INCLUDE_DIRS_MACOS


setup(
    ext_modules=[
        Extension(
            name="hf",
            sources=[
                "./gaussian_basis/gaussian_basis/src/extension.cpp",
                "./gaussian_basis/gaussian_basis/src/basis_function.cpp",
                "./gaussian_basis/gaussian_basis/src/gaussian1d.cpp",
                "./gaussian_basis/gaussian_basis/src/gaussian3d.cpp",
                "./gaussian_basis/gaussian_basis/src/integrals1d.cpp",
                "./gaussian_basis/gaussian_basis/src/integrals3d.cpp",
                "./gaussian_basis/gaussian_basis/src/matrices.cpp",
                "./gaussian_basis/gaussian_basis/src/vec3.cpp"
            ],
            include_dirs=include_dirs,
            library_dirs=lib_dirs,
            extra_compile_args=["-std=c++14", "-O3", "-ffast-math"],
            language="c++"
        )
    ]
)

files = glob.glob("./build/**", recursive=True)
for f in files:
    if f.endswith(".so"):
        shutil.copy(f, "./gaussian_basis/gaussian_basis/extension.so")

shutil.rmtree("./build")
