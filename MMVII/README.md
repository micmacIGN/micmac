# MicMac v2 (MMVII)

**Table of Contents**
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
	- [Linux Ubuntu distribution](#linux-ubuntu-distribution)
 	- [Windows](#windows)
 		- [Install PROJ via vcpkg](#install-proj-via-vcpkg)
 		- [Install MMVII](#install-mmvii)
	- [macOS](#macos)
	- [Additionnal notes](#additionnal-notes)
		- [Compilation details](#compilation-details)
		- [MMVII Command Completion](#mmvii-command-completion)
		- [Graphical User Interface vMMVII](#graphical-user-interface-vmmvii)
		- [Documentation](#documentation)
			- [HTML Documentation](#html-documentation)
			- [PDF Documentation](#pdf-documentation)
- [Run a test](#run-a-test)
- [License](#license)
- [Useful links](#useful-links)

# Description
**MicMac** is a free open-source photogrammetry solution developed at (**[IGN](https://www.ign.fr/)**) - French Mapping Agency - since 2003. A second version named **MMVII** aimed at facilitating external contributions and being more maintainable in the long term has been in development since 2020.

# Prerequisites
Some external tools need to be present on your system for **MMVII** to run properly:
- **[Git](https://git-scm.com/)** to clone the repository
- **[CMake](https://cmake.org/)** to generate build files
- **[make](http://www.gnu.org/software/make)** for parallel processes management
- **[PROJ](http://trac.osgeo.org/proj/)** for coordinate system conversion and coordinate reference system transformation
- **[PROJ additional data](https://download.osgeo.org/proj/)** grids for coordinates tranformations (optional, see doc)
- **[ccache](https://ccache.dev/)** for recompilation optimization (optional)
- **[OpenMP](https://www.openmp.org/)** multi-platform parallel programming (optionnal)
- **[Doxygen](https://www.doxygen.nl/)** documentation generator (optional)
- **[vcpkg](https://github.com/microsoft/vcpkg/blob/master/README.md)** C/C++ library manager (**Windows only**)

# Installation
This section covers the compilation of **MMVII** source code to generate binaries. Pre-compiled binaries are available **[HERE](https://github.com/micmacIGN/micmac/releases)**.

Compilation procedure is discribed below for the 3 main operating systems:
- **[Linux Ubuntu distribution](#linux-ubuntu-distribution)**
- **[Windows](#windows)**
- **[macOS](#macos)** 

## Linux Ubuntu distribution
Before starting the installation, it is necessary to install **MicMac v1** by following the instructions **[HERE](../README.md)**.

Under Linux (Ubuntu) distribution the installation procedure is as follows:

- Open a terminal

- Install dependancies specific to MMVII:
	```bash
	sudo apt install pkg-config libproj-dev
	```

- Access the folder:
	```bash
	cd micmac/MMVII
	```
- Create a directory for building intermediate files and access it:
	```bash
	mkdir build && cd build
	```
- Configure CMAKE and generate makefiles:
	```bash
	cmake ../
	```
- Compile **MMVII**:
	```bash
	make full -j N
	```
	- N is the number of CPUs on the machine and can be retrieved by typing `nproc --all`

- Add binaries to the `PATH` (**adapt the path**):
	```bash
	echo 'export PATH=/home/src/micmac/MMVII/bin:$PATH' >> ~/.bashrc
	```

## Windows
Before starting the installation, it is necessary to install **MicMac v1** by following the instructions **[HERE](../README.md)**.

Under Windows the installation procedure is as follows:

### Install vcpkg (if not done for **MicMac v1**)
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
	cd micmac/MMVII
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
	"C:\src\micmac\MMVII\bin"
	```

## macOS

## Additionnal notes
### Linux compilation details
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

### MMVII Command Completion
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
- Add the following line to your `${HOME}/.bashrc` file:

    ```sh
    [ -f ${HOME}/@MICMAC_SOURCE_DIR@/micmac/MMVII/bash-completion/mmvii-completion ] && . ${HOME}/@MICMAC_SOURCE_DIR@/micmac/MMVII/bash-completion/mmvii-completion
    ```

After making this modification, the command completion feature will be active in any new terminal session.

**Windows:**

 If you're using bash (installed with git for example) on Windows, completion may also works:

- You must have python >= 3.7 installed somewhere

- Edit your ~/.bash_profile and add: (adapt first 2 lines to your case)
    ```
    MMVII_INSTALL_PATH=/c/micmac/MMVII
    PYTHON_INSTALL_PATH=/c/Python/Python39/
    PATH=${PYTHON_INSTALL_PATH}:${MMVII_INSTALL_PATH}/bin:$PATH
    [ -f ${MMVII_INSTALL_PATH}/bash-completion/mmvii-completion ] && . ${MMVII_INSTALL_PATH}/bash-completion/mmvii-completion
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

### Documentation

The latest version of the documentation can be downloaded directly **[HERE](https://github.com/micmacIGN/micmac/releases/tag/MMVII_Documentation)**.

#### HTML Documentation
- Ensure you have doxygen installed (on Ubuntu, you can use the following command):
	```sh
	sudo apt install doxygen
	```
- Navigate to the MMVII directory:
	```sh
	cd micmac/MMVII
	```
- Run the following command:
	```sh
	doxygen Doxyfile
	```

#### PDF Documentation
- Ensure you have LaTeX installed (on Ubuntu, you can use the following command):
	```sh
	sudo apt install texlive
	```
- Navigate to the `MMVII/Doc` directory:
	```sh
	cd micmac/MMVII/Doc
	```
- Run the following command:
	```sh
	make
	```

# Run a test
- In a terminal type:
	```sh
	MMVII Bench 1
	```

# License
This project is licensed under the **CECILL-B** License - see the **[LICENSE.md](LICENSE.md)** file for details.

# Useful links
* [MMVII Documentation](https://github.com/micmacIGN/micmac/releases/tag/MMVII_Documentation)
* [MMVII Programming Session 22-24 Nov 2023](https://www.youtube.com/playlist?list=PLO_lg_3H3aFuMamUsImMzNGPwfkAZge5m)

