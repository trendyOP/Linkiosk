@echo off
setlocal

:: Create build directory if it doesn't exist
if not exist "build" mkdir build
if not exist "bin" mkdir bin

:: Navigate to build directory
cd build

:: Configure with CMake
cmake -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%CD%\..\bin" ..

:: Build and install the project
cmake --build . --target install --config Release

:: Return to original directory
cd ..

echo Build completed.
