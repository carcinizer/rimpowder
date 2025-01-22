# Very simple physics simulator

## Build & run
### Cmake setup
To start development in cmake below command to setup submodules:

```
git submodule update --init --recursive
```

This project uses [cuda-samples](https://github.com/NVIDIA/cuda-samples) repository, originally loated outside this repository in relative location ```../cuda-samples```. Be sure to setup it that way.

then you can configure and build the project:

```
# configure
cmake -B build

#build
(cd build/ && cmake --build . )
```

### Visual Studio setup

```vs2022``` directory contains the solution for this project. The visual studio project requires the SDL3.lib to run the programme. The archive can be downloaded [here](https://github.com/libsdl-org/SDL/releases/tag/preview-3.1.6). The package that was used for development is [SDL3-devel-3.1.6-VC.zip](https://github.com/libsdl-org/SDL/releases/download/preview-3.1.6/SDL3-devel-3.1.6-VC.zip). Unpack the archive in the project's parent directory in ```sdl_vs_lib``` directory.

Correct placement of directories for this project:
```
[project parent dir]
├cuda-samples
├rimpowder
└sdl_vs_lib/
    └── SDL3
        ├── [...]
        └── lib
            ├── arm64
            │   └── [*]
            ├── x64
            │   ├── SDL3.dll
            │   ├── SDL3.lib
            │   ├── SDL3.pdb
            │   └── SDL3_test.lib
            └── x86
                └── [*]
```

