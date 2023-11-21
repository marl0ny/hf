SHELL = /bin/bash

PYTHON_VERSION = 3.10
INCLUDE = -I/usr/include/python${PYTHON_VERSION}\
	  -I${HOME}/.local/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include/numpy\
          -Wno-unused-result -Wsign-compare -g -fopenmp\
		    -fstack-protector-strong\
		  -Wformat -Werror=format-security  -fPIC -DNDEBUG -g -fwrapv \
		  -O3 -Wall -Wstrict-prototypes 
LIBS = -L/usr/lib/python${PYTHON_VERSION}/config-3.8-x86_64-linux-gnu \
       -L/usr/lib -L${HOME}/.local/lib/python${PYTHON_VERSION}/site-packages/numpy/core/lib/\
	   -lpython${PYTHON_VERSION} -lpthread -ldl  -lutil -lm -lnpymath\
	   -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions -fopenmp\

C_COMPILE = gcc# clang 
CPP_COMPILE = g++ -std=c++14

# For MacOS you must first install the relavent OpenGL and GLFW libraries.
# Depending on which package manager you used, as well as the age and 
# type of your system, the actual location of these libraries will vary.
# You must first determine the actual location of the developement directories
# for your own system, and then edit the INCLUDE and LIBS 
# variables accordingly.
ifeq ($(shell uname),Darwin)
# INCLUDE =  -I${PWD} -I${PWD}/gl_wrappers -I/opt/homebrew/include\
# 					 -I/usr/local/include
# LIBS = -ldl -L/opt/homebrew/lib -L/usr/local/include -lglfw\
#        -framework CoreVideo -framework OpenGL -framework IOKit\
#        -framework Cocoa -framework Carbon -framework Accelerate\
# 	   -lpng
# C_COMPILE = /opt/homebrew/opt/llvm/bin/clang 
# CPP_COMPILE = /opt/homebrew/opt/llvm/bin/clang++ -std=c++14
# -I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/llvm/include/c++/v1
# -L/opt/homebrew/opt/llvm/lib
# -export-dynamic
#-Wl,-Bsymbolic-functions
# -Wl,-O3
INCLUDE = -I/opt/homebrew/include\
		-I/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/include/python3.9\
		-I/Users/mark/Library/Python/3.9/lib/python/site-packages/numpy/core/include/numpy\
		-Wno-unused-result -Wsign-compare -g -Xpreprocessor -fopenmp\
		-fstack-protector-strong\
		-Wformat -Werror=format-security  -fPIC -DNDEBUG -g -fwrapv \
		-O3 -Wall -Wstrict-prototypes 
LIBS = -L/opt/homebrew/lib\
		-L/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib\
		-L/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/lib-dynload \
		-L/usr/lib -L/Users/mark/Library/Python/3.9/lib/python/site-packages/numpy/core/lib\
		-lpython3.9 -lpthread -ldl  -lutil -lm -lnpymath\
		# -Xlinker -lomp
endif

TARGET = ../extension.so
FLAGS = -O3 -g -ffast-math -Wall # -Wno-everything
# FLAGS = -O0 -g -Wall

CPP_SOURCES = vec3.cpp gaussian1d.cpp gaussian3d.cpp \
				integrals1d.cpp integrals3d.cpp matrices.cpp \
				extension.cpp
CPP_OBJECTS = vec3.o gaussian1d.o gaussian3d.o integrals1d.o \
				integrals3d.o matrices.o extension.o

all: ${TARGET}

${TARGET}: ${CPP_OBJECTS}
	${CPP_COMPILE} ${FLAGS} -shared -o $@ $^ ${LIBS}

${CPP_OBJECTS}: ${CPP_SOURCES}
	${CPP_COMPILE} ${FLAGS} -c $^ ${INCLUDE}

clean:
	rm -f *.o ${TARGET}
