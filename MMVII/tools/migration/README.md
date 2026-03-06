# MicMac v2 (MMVII)

**Table of Contents**
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Installation from sources (Linux/Windows)](#installation-from-sources)
	- [Linux Ubuntu distribution](#linux-ubuntu-distribution)
 	- [Windows](#windows)
	- [Additionnal notes](#additionnal-notes)
		- [Compilation details](#compilation-details)
		- [Compilation with MicMac V1 API](#compilation-with-micmac-v1-api)
		- [Graphical User Interface vMMVII](#graphical-user-interface-vmmvii)
- [Installation from binary (Windows only)](#installation-from-binary-windows-only)
- [Run a test](#run-a-test)
- [Documentation](#documentation)
- [MMVII Command Completion](#mmvii-command-completion)
- [License](#license)
- [Useful links](#useful-links)


# Description
**[MicMac](https://github.com/micmacIGN/micmac)** is a free open-source photogrammetry solution developed at (**[IGN](https://www.ign.fr/)**) - French Mapping Agency - since 2003. 
This repository contains the second version named **MMVII** aimed at facilitating external contributions and being more maintainable in the long term has been in development since 2020.

# Prerequisites
Compilation tools need to be present on your system to build **MMVII** properly:
- **C++ compiler (g++ or clang on Linux, MSVC++ on Windows)**
- **[Git](https://git-scm.com/)** to clone the repository
- **[CMake](https://cmake.org/)** to generate build files
- **[make or ninbja](http://www.gnu.org/software/make)** to build executable (**Linux only**)
- **[ccache](https://ccache.dev/)** for recompilation optimization (**Linux only** optional)
- **[vcpkg](https://github.com/microsoft/vcpkg/blob/master/README.md)** C/C++ library manager (**Windows only**)
- **[OpenMP](https://www.openmp.org/)** multi-platform parallel programming (optionnal)
- **[Doxygen](https://www.doxygen.nl/)** documentation generator (optional)

Some external libraries need to be present on your system (installation is described bellow for each platform):
- **[PROJ](http://trac.osgeo.org/proj/)** for coordinate system conversion and coordinate reference system transformation
- **[PROJ additional data](https://download.osgeo.org/proj/)** grids for coordinates tranformations (optional, see doc)
- **[GDAL](https://gdal.org/)** for image files handling

# Installation from sources
This section covers the compilation of **MMVII** source code to generate binaries.\
(Pre-compiled binaries for Windows are available **[HERE](https://github.com/micmacIGN/micmac/releases)**)

Some specific MMVII commands may require the MicMac V1 binary and will complain if it is not found.
In this case, install [micmac v1](https://github.com/micmacIGN/micmac) and make sure that **micmac/bin** is in you PATH environment variable.

Compilation procedure is described below for:
- **[Linux Ubuntu distribution](#linux-ubuntu-distribution)**
- **[Windows](#windows)**



## Linux Ubuntu distribution

Under Linux (Ubuntu) distribution the installation procedure is as follows:

- Open a terminal

- Install dependancies specific to MMVII:
	```bash
	sudo apt install pkg-config libproj-dev libgdal-dev libxerces-c-dev
	```

- Access the folder:
	```bash
	cd MMVII
	```
- Create a directory for building intermediate files and access it:
	```bash
	mkdir build && cd build
	```
- Configure CMAKE and generate makefiles:
	```bash
	cmake ..
	```
- Compile **MMVII**:
	```bash
	make full -j8   
	```
	- in general, you can run "make -jNUM"  where NUM is the number of CPUs on the machine and can be retrieved by typing `nproc --all`, or use "NUM-2" if you want to use the computer while compiling
        

- Add binaries to the `PATH` (**adapt the path**):
	```bash
	echo 'export PATH=/home/src/MMVII/bin:$PATH' >> ~/.bashrc
	```

## Windows
Under Windows the installation procedure is as follows:

### Install vcpkg 
- Open a **Git Bash** terminal
- In another working directory, clone the repository:
	```bash
	git clone  https://github.com/microsoft/vcpkg.git
	```
- Access the folder:
	```bash
	cd vcpkg
	```
- Setup vcpkg:
	```bash
	./bootstrap-vcpkg.bat
	```
	```bash
	vcpkg.exe integrate install
	```
	
### Install MMVII
- Open a **Git Bash** terminal
- Access the folder:
	```bash
	cd MMVII
	```
- Create a directory for building intermediate files and access it:
	```bash
	mkdir build && cd build
	```
- Configure cmake and generate Makefiles:
	```bash
	"[CMAKE_DIR]/cmake.exe" .. "-DCMAKE_TOOLCHAIN_FILE=[VCPKG_DIR]/vcpkg/scripts/buildsystems/vcpkg.cmake"
	```
- Compile **MMVII**:
	```bash
	"[CMAKE_DIR]/cmake.exe" --build . --target full --config Release
	```
- Add binaries to Windows `PATH` environment variable via **Advanced system settings** menu. Example of path (**adapt the path**):
	```bash
	"C:\src\MMVII\bin"
	```

## Additionnal notes
### Compilation details
- If using CLang version XX and want OpenMP: `sudo apt install libomp-XX-dev`
- You can use `cmake -G Ninja ..` to use Ninja build system instead of the native one. (`sudo apt install ninja-build`)
- Use `cmake --build . -j N` or `cmake --build . -j N -v` instead of make (works with all build systems)
- Use `cmake --build . --target clean` or `cmake --build . --target cleanall`
- Use `ccmake ..` or `cmake-gui ..` to change config option:
- CMAKE_BUILD_TYPE:
	- Debug : -g
	- RelWithDebInfo : -O3 -g  (default)
	- Release : -O3 -DNDEBUG
- CMAKE_CXX_COMPILER (advanced mode : 't'):
	- Allow to set compiler version and type (g++, clang)
- Clean :
	- make clean     : delete build products
	- make distclean : delete build products and generated codes
	- rm -fr MMVII/build/* : reinitialize the build configuration


### Compilation with micmac V1 API
MMVII does not use **MicMac v1** anymore, so installing **MicMac V1** is not required.
However some features of MMVII still require calls to the MicMac v1 library and have not yet been rewritten in MMVII. They are disabled. 
For those who really need it, you can reactivate use of the MicMac V1 lib  :

- Install **MicMac v1** by following the instructions **[HERE](https://github.com/micmacIGN/micmac)**.

- Activate the CMake option **MMVII_KEEP_LIBRARY_MMV1** in the step '__Configure CMAKE and generate makefiles:
        ```bash
        cmake .. -DMMVII_KEEP_LIBRARY_MMV1=on -DMMV1_PATH=your_directory_of_micmacv1
        ```


### Graphical User Interface vMMVII
The **vMMVII** tool provides a convenient graphical user interface (GUI) for writing **MMVII** commands.
To compile it, add "**-DvMMVII_BUILD=ON**" on then cmake configure command line.

For Ubuntu 22.04, you can install the necessary QT5 package with the following command:
```bash
sudo apt install qtbase5-dev
```
For windows, it will be automatically downloaded and compiled (may take a very long time the first time)

To use **vMMVII**, simply type `vMMVII` in a terminal in your working directory.

Please note:
- Currently, there is no documentation available.
- The tool is in beta, so some MMVII parameters may be misinterpreted or may not have the appropriate File Dialog helper.


# Installation from binary (Windows only)
**WARNING**: MMVII is essentially a command line tool with a somewhat specific syntax.


Download the MMVII archive file [here](https://github.com/micmacv2/MMVII/releases/download/Windows_MMVII_build/mmvii_windows.zip).\
Extract the .zip file in the directory of you choice (avoid **c:\Program Files**), c:\pgms for example.\
The main executable will be  **c:\pgms\MMV2\bin\MMVII.EXE**. There is a graphical front-end to help writing command line: **c:\pgms\MMV2\bin\vMMVII.EXE**.\


You can add the MMVI\bin path (in this example, c:\pgms\MMVII\bin) to your environment PATH variable.


# Run a test
- In a terminal type:
	```sh
	MMVII Bench 1
	```
There may be a lot of cryptic messages and some **"##   - Nb Warning "** at the end, but the test passed if execution **does not** end with a message of the form:
```
	######################################
	Level=[UserEr:xxxxxxx]
	Mes=[xxxxxxx xxxxxxxxxx xxxxxx xxxxxx]
	========= ARGS OF COMMAND ==========
	C:\pgms\MMVII\bin\MMVII.exe Bench 1	
```
	

# Documentation

The latest version of the (work in progress) documentation can be downloaded directly **[HERE](https://github.com/micmacv2/MMVII/releases/download/MMVII_Documentation/Doc2007_a4.pdf)**.
You can build documentation from sources if you have installed the MMVII sources:


### Building Doxygen HTML documentation
- Ensure you have doxygen installed (on Ubuntu, you can use the following command):
	```sh
	sudo apt install doxygen
	```
- Navigate to the MMVII directory:
	```sh
	cd MMVII
	```
- Run the following command:
	```sh
	doxygen Doxyfile
	```

### Building PDF documentation
- Ensure you have LaTeX installed (on Ubuntu, you can use the following command):
	```sh
	sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-science
	```
- Navigate to the `MMVII/Doc` directory:
	```sh
	cd MMVII/Doc
	```
- Run the following command:
	```sh
	make
	```
	
# MMVII Command Completion
You can enable command completion for **MMVII** in Linux Bash, which simplifies the use of **MMVII** commands.

**Requirements:**
- `bash-completion`
- `python-is-python3`

These are typically installed by default on Ubuntu. If not, you can install them using:
	```bash
	sudo apt install bash-completion python-is-python3`
	```

**Configuration:**
- Ensure MMVII is compiled.
- Make sure the MMVII executable is in your `$PATH`.
- Add the following line to your `${HOME}/.bashrc` file (replace __@MICMAC_SOURCE_DIR@__ with the directectory where your MMVII directory is)

    ```sh
    [ -f ${HOME}/@MICMAC_SOURCE_DIR@/MMVII/bash-completion/mmvii-completion ] && . ${HOME}/@MICMAC_SOURCE_DIR@/MMVII/bash-completion/mmvii-completion
    ```

After making this modification, the command completion feature will be active in any **new** terminal session.

**Windows:**

 If you're using bash (installed with git for example) on Windows, completion may also works:

- You must have python >= 3.7 installed somewhere

- Edit your ~/.bash_profile and add: (adapt first 2 lines to your case)
    ```
    MMVII_INSTALL_PATH=/c/src/MMVII
    PYTHON_INSTALL_PATH=/c/Python/Python39/
    PATH=${PYTHON_INSTALL_PATH}:${MMVII_INSTALL_PATH}/bin:$PATH
    [ -f ${MMVII_INSTALL_PATH}/bash-completion/mmvii-completion ] && . ${MMVII_INSTALL_PATH}/bash-completion/mmvii-completion
	```

# License
This project is licensed under the **CECILL-B** License - see the **[LICENSE.md](LICENSE.md)** file for details.

MMVII sources includes codes from:

 - hapPLY: Copyright (c) 2018 Nick Sharp, MIT licence, https://github.com/nmwsharp/happly
 - Delaunay/delaunator: Copyright (c) 2018 Volodymyr Bilonenko, MIT Licence
 - Eigen: Copyright (C) 2008 Gael Guennebaud, Mozilla Public License, https://eigen.tuxfamily.org
 - libE57Format, Copyright (C) 2020 Andy Maloney/Kevin Ackley, Boost Software License, https://github.com/asmaloney/libE57Format


# Useful links
* [MicMac v1](https://github.com/micmacIGN/micmac)
* [MMVII Documentation](https://github.com/micmacv2/MMVII/releases/tag/MMVII_Documentation)
* [MMVII Programming Session 22-24 Nov 2023](https://www.youtube.com/playlist?list=PLO_lg_3H3aFuMamUsImMzNGPwfkAZge5m)

