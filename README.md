# MicMac

- For **MicMac v2 (MMVII)** click **[HERE](MMVII/README.md)**. 

**Table of Contents**
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
	- [Linux Ubuntu distribution](#linux-ubuntu-distribution)
 	- [Windows](#windows)
	- [macOS](#macos)
	- [Additionnal notes](#additionnal-notes)
		- [Install Homebrew Package Manager for macOS](#install-homebrew-package-manager-for-macos)
		- [Qt Tools](#qt-tools)
			- [Issues](#issues)
		- [PATH and pre-compiled binaries ](#path-and-pre-compiled-binaries)
		- [MicMac via a Docker image](#micmac-via-a-docker-image)
- [Run the example dataset](#run-the-example-dataset)
- [License](#license)
- [Useful links](#useful-links)


# Description
**MicMac** is a free open-source photogrammetric software for 3D reconstruction under development at the National Institute of Geographic and Forestry Information - French Mapping Agency - (**[IGN](https://www.ign.fr/)**) and the National School of Geographic Sciences (**[ENSG](https://ensg.eu/)**) withing the **[LASTIG](https://www.umr-lastig.fr/)** lab. **MicMac** is distributed under **[CECILL-B](LICENSE.md)** license since 2007.

# Prerequisites
Some external tools need to be present on your system for **MicMac** to run properly:
- **[Git](https://git-scm.com/)** to clone the repository
- **[CMake](https://cmake.org/)** to generate build files
- **[make](http://www.gnu.org/software/make)** for parallel processes management
- **[ImageMagick](http://www.imagemagick.org)** for image format conversion
- **[exiftool](http://www.sno.phy.queensu.ca/~phil/exiftool)** and **[exiv2](http://www.exiv2.org)** to read/write image meta-data
- **[PROJ](http://trac.osgeo.org/proj/)** for coordinate system conversion and coordinate reference system transformation
- **[Xlib](https://gitlab.freedesktop.org/xorg/lib/libx11)** to launch some GUI tools based on X window system
- **[ccache](https://ccache.dev/)** for recompilation optimization (optional)
- **[Qt](https://www.qt.io/)** to launch some GUI tool based on QT (optionnal)

# Installation
This section covers the compilation of **MicMac** source code to generate binaries. Pre-compiled binaries are available **[HERE](https://github.com/micmacIGN/micmac/releases)**.

Compilation procedure is discribed below for the 3 main operating systems:
- **[Linux Ubuntu distribution](#linux-ubuntu-distribution)**
- **[Windows 10](#windows-10)**
- **[macOS](#macos)** 


## Linux Ubuntu distribution
Under Linux (Ubuntu) distribution the installation procedure is as follows:

- Open a terminal
- Install dependencies:
	```bash
	sudo apt-get install git cmake make ccache imagemagick libimage-exiftool-perl exiv2 proj-bin libx11-dev qt5-default
	```
- Clone the repository:
	```bash
	git clone https://github.com/micmacIGN/micmac.git
	```
- Access the folder:
	```bash
	cd micmac
	```
- Create a directory for building intermediate files and access it:
	```bash
	mkdir build && cd build
	```
- Generate makefiles:
	```bash
	cmake ..
	```
- Compile:
	```bash
	make install -j N
	```
	- N is the number of CPUs on the machine and can be retrieved by typing `nproc --all`

- Add binaries to the `PATH` (**adapt the path**):
	```bash
	echo 'export PATH=/home/src/micmac/bin:$PATH' >> ~/.bashrc
	```

## Windows
Under Windows the installation procedure is as follows:
- Download and Install **[Build Tools for Visual Studio](https://visualstudio.microsoft.com/)**
- Download and Install **[Git](https://git-scm.com/)**
- Download and Install **[CMake](https://cmake.org/)**. Make sure cmake.exe is in the %PATH%
- Open a **Git Bash** terminal
- Optionnal, QT5 tools : Download and Install **[vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started)** 
  in a general directory (c:\pgms, for example):
   ```bash
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg
   bootstrap-vcpkg.bat
   vcpkg.exe integrate install

   ```
- Clone the repository:
	```sh
	git clone https://github.com/micmacIGN/micmac.git
	```
- Access **micmac** folder:
	```bash
	cd micmac
	```
- Create a directory for building intermediate files and access it:
	```bash
	mkdir build && cd build
	```
- Generate Microsoft Visual Studio Solution File **MICMAC.sln**:

	- Without Qt5 Tools:
	```bash
	cmake.exe ..
	```
    - With Qt5 Tools (This will download and compile QT5, it will take a very long time):
    ```bash
	cmake .. -DWITH_QT5=1 -DCMAKE_TOOLCHAIN_FILE=c:/pgms/vcpkg/script/buildsystem/vcpkg.cmake
	```
- Compile **MicMac**:
	```bash
	cmake.exe" --build . --config Release --target INSTALL
	```
- Add binaries to Windows `PATH` environment variable via **Advanced system settings** menu. Example of path (**adapt the path**):
	```bash
	"C:\src\micmac\bin"
	```

## macOS
Under macOS we will use **[Homebrew](https://brew.sh/)** Package Manager to install dependencies.

If you don't have Homebrew, first follow the instructions **[HERE](#install-homebrew-package-manager-for-macos)**. 

Under macOS the installation procedure is as follows:

- Open a terminal
- Use **Homebrew** to install dependencies:
	```bash
	brew install git
	brew install cmake
	brew install imagemagick
	brew install exiftool
	brew install exiv2
	brew install proj
	brew install qt5
	```
- Clone the repository:
	```bash
	git clone https://github.com/micmacIGN/micmac.git
	```
- Access the folder:
	```bash
	cd micmac
	```
- Create a directory for building intermediate files and access it:
	```bash
	mkdir build && cd build
	```
- Generate makefiles:
	```bash
	cmake ..
	```
- Compile **MicMac**:
	```bash
	make install -j N
	```
	- N is the number of CPUs on the machine

- Add binaries to the `PATH` (**adapt the path**):
	```bash
	echo 'export PATH=/home/src/micmac/bin:$PATH' >> ~/.zshrc
	```

## Additionnal notes

### Install Homebrew Package Manager for macOS
- Open a terminal
- Download and run Homebrew installation script:
	```bash
	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	```
- Follow the on-screen instructions to complete the installation
- Add **Homebrew** to the configuration file of the Zsh shell environment:
	```bash
	echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
	```
- Execute the commands from the `.zshrc` file in the current shell session:
	```bash
	source ~/.zshrc
	```
- Check installation:
	```bash
	brew doctor
	```

### Qt Tools (Linux, MacOS)
To use Qt GUIs, you need to adapt the **cmake** command as follows:
	```bash
	cmake ../ -DWITH_QT5=1
	```

#### Issues
- In case **cmake** complains about missing Widgets library, you must assign `CMAKE_PREFIX_PATH` variable:
	```bash
	cmake ../ -DWITH_QT5=1 -DCMAKE_PREFIX_PATH=path/to/qt/X.XX.X/ 
	```
- For Linux/macOS it is sometimes necessary to append the `lib` directory to `LD_LIBRARY_PATH` in `.bashrc` / `.zshrc` to be able to use Qt tools:

	- Under Linux :
		```bash 
		echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/src/micmac/lib/' >>  ~/.bashrc
		```
	- Under macOS:
		```bash 
		echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/src/micmac/lib/' >>  ~/.zshrc
		```

### PATH and pre-compiled binaries 
You can append the full path of the `bin` directory to `PATH` environment variable to call **MicMac** commands from anywhere. However, it is not necessary to add the `binaire-aux` directory to the `PATH` variable.

### MicMac via a Docker image
A precompiled docker image is available and ready to use:
	```sh
	docker pull rupnike/micmac
	```
or build your own image from scratch using the existing Dockerfile:
	```sh
	docker image build -t micmac:1.0 -f Dockerfile
	```
[![Docker Status](https://dockeri.co/image/rupnike/micmac)](https://hub.docker.com/r/rupnike/micmac/)

# Run the example dataset
- Download the test dataset available **[HERE](https://micmac.ensg.eu/data/gravillons_dataset.zip)**
- unzip the folder and open a terminal inside the folder containing the images
- Run processing scripts:
	- under Linux (Ubuntu) distribution:
		```sh
		sh gravillons_test.sh
		```
 	- under Windows:
		```sh
		./gravillons_test.bat
		```
# License
This project is licensed under the **CECILL-B** License - see the **[LICENSE.md](LICENSE.md)** file for details.

# Useful links
* [MicMac Documentation](https://github.com/micmacIGN/Documentation/blob/master/DocMicMac.pdf)
* [MicMac Wiki](https://micmac.ensg.eu/)
* [MicMac Reddit](https://www.reddit.com/r/MicMac/)
* [MicMac Sketchfab](https://sketchfab.com/micmac)
