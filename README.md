MICMAC
======
[![Build Status](https://travis-ci.org/micmacIGN/micmac.svg?branch=master)](https://travis-ci.org/micmacIGN/micmac)

[Version française](LISEZMOI.md)

# Prerequisites

Some external tools need to be present on your system for Micmac to run properly :
- [make](http://www.gnu.org/software/make) for parallel processes management,
- *convert*, from [ImageMagick](http://www.imagemagick.org), for image format conversion,
- [exiftool](http://www.sno.phy.queensu.ca/~phil/exiftool) and [exiv2](http://www.exiv2.org), to read/write image meta-data,
- [proj4](http://trac.osgeo.org/proj/) for coordinate system conversion.

On Debian/Ubuntu distribution you can easily install these tools by calling this command:

`sudo apt-get install make imagemagick libimage-exiftool-perl exiv2 proj-bin qt5-default`

You can check before-hand that Micmac is able to find those programs by calling the command:

`bin/mm3d CheckDependencies` (in Micmac directory)

*NOT FOUND* near one of the tools indicates either the specified executable is not on your disk or it cannot be found in the
directories of the *PATH* environment variable.
There is also a special directory for tool finding which is named *binaire-aux*, in Micmac directory. When an external program
is needed, this directory is always scanned whatever the value of PATH.

## Additionnal notes for Windows

You will need [Visual C++ 2010 runtime redistribuables](http://www.microsoft.com/fr-fr/download/details.aspx?id=5555) to run pre-compiled binaries of micmac.
Both pre-compiled and compiled from source executables will need :
- [Visual C++ 2005 runtime redistribuables](http://www.microsoft.com/fr-fr/download/details.aspx?id=3387),
- and [Net Framework 2.0](http://www.microsoft.com/fr-fr/download/details.aspx?id=1639).

One of *WINDIR* or *SystemRoot* environment variable must be set to Windows' installation directory (`C:\Windows` in much cases).
This prevents Micmac from calling a `convert.exe` that is not *ImageMagick's* convert tool but a network utility used by the system.
Since Windows does not have an easy-to-use package manager, a version of *make*, *convert*, *exiftool* and *exiv2* are delivered with
the source and Windows binaries archives. They are placed in the `binaire-aux` directory.

# Compiling from the sources' archive

## Prerequisites

In addition of previously named tools, people willing to compile binaries from the source code will need to install the [cmake](www.cmake.org)
program. Linux and MacOS X users may also want to get X11 header files in order to generate graphical functionalities like *SaisieBasc*, *SaisieMasque*, etc ...
The package of X11 headers is general called `libx11-dev` under Linux distributions.
X11-based tools are not available in the Windows version.
Windows users may need Qt5 libraries to generate graphical interfaces such as *SaisieMasqQT*.

For recompilation optimization, [ccache](ccache.dev) is automatically used if detected.

## Compiling process for Linux / MacOS X

- clone the git repository : `git clone https://github.com/micmacIGN/micmac.git`
- enter 'micmac' directory : `cd micmac`
- create a directory for the build's intermediate files, then enter it : `mkdir build & cd build`
- generate makefiles using cmake : `cmake ../`
- process compilation : `make install -j*cores number*` (ex.: `make install -j4`)

## Compiling process for Visual C++ (Windows)

The first steps are the same as for a Linux/MacOS build except for the `make` call.
Instead of makefiles, *Cmake* generates a Visual C++ solution, named `micmac.sln`. Open it and compile the `INSTALL` project. 
Be sure to be in *Release* configuration, for Micmac is much faster built this way than in *Debug* mode.
Again, do not compile the entire solution but just the `INSTALL` project, otherwise compiled binaries won't be copied in the `bin` directory and this will prevent Micmac from working.

## Docker image
A precompiled docker image is available and ready to use:

`docker pull rupnike/micmac`

or build your own image from scratch using the existing Dockerfile:

`docker image build -t micmac:1.0 -f Dockerfile`

[![Docker Status](https://dockeri.co/image/rupnike/micmac)](https://hub.docker.com/r/rupnike/micmac/)

## Install MicMac in WinOS subsystem

You can also use MicMac on Windows 10 through the Windows Subsystem for Linux (WSL). WSL allows you to run a Linux distribution (e.g. Ubuntu) directly on Windows, unmodified, without the overhead of a traditional virtual machine or dualboot setup. For further information please refer to the instructions in this [WSL tutorial](https://micmac.ensg.eu/index.php/Install_MicMac_in_Windows_Subsystem_for_Linux).

# Installation test

The website [logiciels.ign.fr](http://logiciels.ign.fr/?Telechargement,20) also provides a test dataset called `Boudha_dataset.zip`.
This file contains images and configuration files needed to compute the *Boudha* example from Micmac's documentation. By calling the script this way :

`./boudha_test.sh my_micmac_bin/`

assuming your working directory is the *Boudha* directory contained in the file, you can process all the tool-chain until dense point matching. 

'my_micmac_bin' is a path to the 'bin' directory of your installation.
	ex.: ./boudha_test.sh ../micmac/bin/
	This example assumes 'Boudha' directory (containing data) and 'micmac' (installation directory) have the same parent directory. Notice
the ending '/', it's mandatory for the script to work.
	After some computation time, you may find three 'ply' files in the 'MEC-6-Im' directory with the three parts of the dense points cloud
of the statue's head. Open the PLY files with a viewer like meshlab to check everything proceeded correctly.

# Additionnal notes

You can append the full path of the `bin` directory to `PATH` environment variable to call Micmac commands from anywhere. However, it is not necessary to add the `binaire-aux` directory to the `PATH` variable.

For Linux / MacOSX, you have to append the path to the `lib` directory to `LD_LIBRARY_PATH` in `.bashrc` to be able to use QT tools. 
Add the following line: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/micmac/lib/`

For MacOSX, if you want to use QT tools with precompiled binaries available on [logiciels.ign.fr](http://logiciels.ign.fr/?Telechargement,20), you need to install Qt libraries for Mac from [http://download.qt-project.org](http://download.qt-project.org/archive/qt/4.8/4.8.4/qt-opensource-mac-4.8.4.dmg)
