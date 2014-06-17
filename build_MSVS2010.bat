call "%PROGRAMFILES%\Microsoft Visual Studio 10.0\VC\vcvarsall.bat"

mkdir build
cd build
cmake -G "NMake Makefiles" ..
nmake install
cd ..
