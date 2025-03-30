# constrained-pnp-writeup

## Building

### Python

Install requirements.txt and run [src/test.py](src/test.py)

### C++

```
git submodule update --init --recursive
cmake -B build -G Ninja
cmake --build build
cmake --build build --target test
```
