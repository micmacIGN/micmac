# MicMac

- For the french version click **[HERE](LISEZMOI.md)**.
- For **MicMac v2 (MMVII)** click **[HERE](MMVII/Readme.md)**. 

**Table of Contents**
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
	- [Linux Ubuntu distribution](#linux-ubuntu-distribution)
 	- [Windows 10](#windows-10)
		- [Windows via WSL subsystem](#windows-via-wsl-subsystem)
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
**MicMac** is a free open-source photogrammetric software for 3D reconstruction under development at the National Institute of Geographic and Forestry Information - French Mapping Agency - (**[IGN](https://www.ign.fr/)**) and the National School of Geographic Sciences (**[ENSG](https://ensg.eu/)**). **MicMac** is distributed under **[CECILL-B](LICENSE.md)** license since 2007.

# Prerequisites
Some external tools need to be present on your system for **MicMac** to run properly :
- **[Git](https://git-scm.com/)** to clone the repository
- **[CMake](https://cmake.org/)** to generate build files
- **[make](http://www.gnu.org/software/make)** for parallel processes management
- **[ccache](https://ccache.dev/)** for recompilation optimization (optional)
- **[ImageMagick](http://www.imagemagick.org)** for image format conversion
- **[exiftool](http://www.sno.phy.queensu.ca/~phil/exiftool)** and **[exiv2](http://www.exiv2.org)** to read/write image meta-data
- **[PROJ](http://trac.osgeo.org/proj/)** for coordinate system conversion and coordinate reference system transformation
- **[Xlib](https://gitlab.freedesktop.org/xorg/lib/libx11)** to launch some GUI tools based on X window system
- **[Qt](https://www.qt.io/)** to launch some GUI tool based on QT (optionnal)

# Installation
This section covers the compilation of **MicMac** source code to generate binaries. Pre-compiled binaries are available **[HERE](https://github.com/micmacIGN/micmac/releases)**.

Compilation procedure is discribed below for the 3 main operating systems:
- [Linux Ubuntu distribution](#linux-ubuntu-distribution)
- [Windows 10](#windows-10)
- [macOS](#macos) 


## Linux Ubuntu distribution
Under Linux (Ubuntu) distribution the installation procedure is as follows:

1. Open a terminal
2. Installing dependencies:
```bash
sudo apt-get install git cmake make ccache imagemagick libimage-exiftool-perl exiv2 proj-bin libx11-dev qt5-default
```
3. Clone **MicMac** repository:
```bash
git clone https://github.com/micmacIGN/micmac.git
```
4. Access **micmac** folder:
```bash
cd micmac
```
5. Create a directory for building intermediate files and access it:
```bash
mkdir build && cd build
```
6. Generate makefiles:
```bash
cmake ../
```
7. Compile **MicMac**:
```bash
make install -jX
```
- X is the number of CPUs on the machine (the number can be retrieved by typing `nproc --all`)

8. Add **MicMac** to the `PATH` (**adapt the path**):
```bash
echo 'export PATH=/home/src/micmac/bin:$PATH' >> ~/.bashrc
```

## Windows 10
Under Windows the installation procedure is as follows:
1. Download and Install **[Microsoft Visual Studio Community 2019](https://visualstudio.microsoft.com/fr/vs/older-downloads/)** and select Env Dev C++
2. Download and Install **[Git](https://git-scm.com/)**
3. Download and Install **[CMake](https://cmake.org/)**
4. Download and Install **[Qt](https://www.qt.io/)**
5. Open a **Git Bash** terminal
6. Clone **MicMac** repository:
```sh
git clone https://github.com/micmacIGN/micmac.git
```
6. Access **micmac** folder:
```bash
cd micmac
```
7. Create a directory for building intermediate files and access it:
```bash
mkdir build && cd build
```
8. Generate Visual Studion Solution **MICMAC.sln**:
```bash
"C:\Program Files\CMake\bin\cmake.exe" ../
```
9. Compile **MicMac**:
```bash
"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release --target INSTALL
```
10. Add **MicMac** to Windows `PATH` environment variable via **Advanced system settings** menu. Example of path (**adapt the path**):
```bash
"C:\src\micmac\bin"
```

### Windows via WSL subsystem
You can also use MicMac on Windows 10 through the Windows Subsystem for Linux (WSL). WSL allows you to run a Linux distribution (e.g. Ubuntu) directly on Windows, unmodified, without the overhead of a traditional virtual machine or dualboot setup. For further information please refer to the instructions in this **[WSL tutorial](https://micmac.ensg.eu/index.php/Install_MicMac_in_Windows_Subsystem_for_Linux)**.

## macOS
Under macOS we will use **[Homebrew](https://brew.sh/)** Package Manager to install dependencies.

If you don't have Homebrew, first follow the instructions **[HERE](#install-homebrew-package-manager-for-macos)**. 

Under macOS the installation procedure is as follows:

1. Open a terminal
2.  Use **Homebrew** to install dependencies:
```bash
brew install git
brew install cmake
brew install imagemagick
brew install exiftool
brew install exiv2
brew install proj
brew install qt5
```
3. Clone **MicMac** repository:
```bash
git clone https://github.com/micmacIGN/micmac.git
```
4. Access **micmac** folder:
```bash
cd micmac
```
5. Create a directory for building intermediate files and access it:
```bash
mkdir build && cd build
```
6. Generate makefiles:
```bash
cmake ../
```
7. Compile **MicMac**:
```bash
make install -jX
```
- X is the number of CPUs on the machine

8. Add **MicMac** to the `PATH` (**adapt the path**):
```bash
echo 'export PATH=/home/src/micmac/bin:$PATH' >> ~/.zshrc
```

## Additionnal notes

### Install Homebrew Package Manager for macOS
1. Open a terminal
2.  Download and run Homebrew installation script:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Follow the on-screen instructions to complete the installation
4. Add **Homebrew** to the configuration file of the Zsh shell environment:
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
```
5. Execute the commands from the `.zshrc` file in the current shell session:
```bash
source ~/.zshrc
```
6. Check installation:
```bash
brew doctor
```

### Qt Tools
To use Qt GUIs, you need to adapt the **cmake** command as follows:
```bash
cmake ../ -DWITH_QT5=1
```

#### Issues
In case **cmake** complains about missing Widgets library, you must assign `CMAKE_PREFIX_PATH`:
```bash
cmake ../ -DWITH_QT5=1 -DCMAKE_PREFIX_PATH=path/to/qt/X.XX.X/ 
```
For Linux/macOS it is sometimes necessary to append the `lib` directory to `LD_LIBRARY_PATH` in `.bashrc` / `.zshrc` to be able to use Qt tools:

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
1. Download the test dataset available **[HERE](https://micmac.ensg.eu/data/gravillons_dataset.zip)**
2. unzip the folder and open a terminal inside the folder containing the images
3. Run processing scripts:
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
