# Very quick readme

To start development use:

```
git submodule update --init --recursive
```

then you can configure and build the project:

```
# configure
cmake -B d_build -DDUMMY1_OPTION=ON

#build
(cd d_build/ && cmake --build . )
```