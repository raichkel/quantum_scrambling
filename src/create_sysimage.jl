# creates sysimage for project - pre-compiles all required dependencies within project into .so file
# for faster unpacking and running

using PackageCompiler
using Pkg

Pkg.activate(".")

create_sysimage(["PackageCompiler","Plots"]; sysimage_path="src/sysimage.so")