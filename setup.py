import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop as _develop


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name will be used as the extension name, and required to match
# the CMake target name.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent.resolve()  # Ensure extdir is absolute

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        cfg = "Debug" if self.debug else "Release"

        # Set Python_EXECUTABLE, Python_INCLUDE_DIR explicitly.
        python_executable = sys.executable
        python_include_dir = sysconfig.get_path("include")

        # Try to find the Python library file robustly
        python_library_file = None
        libdir = sysconfig.get_config_var("LIBDIR")
        library = sysconfig.get_config_var("LIBRARY")
        if libdir and library:
            candidate = Path(libdir) / library
            if candidate.exists():
                python_library_file = str(candidate.resolve())
            else:
                # Sometimes LIBRARY might include the lib prefix/suffix, sometimes not
                # Try finding based on common patterns if the direct path fails
                potential_lib_names = [
                    f"libpython{sysconfig.get_python_version()}.a",
                    f"libpython{sysconfig.get_python_version()}.dylib",
                    f"python{sys.version_info.major}{sys.version_info.minor}.lib",  # Windows
                ]
                for name in potential_lib_names:
                    candidate = Path(libdir) / name
                    if candidate.exists():
                        python_library_file = str(candidate.resolve())
                        break

        # If still not found, try sysconfig.get_config_var('LIBPL') as last resort directory
        if not python_library_file:
            libpl = sysconfig.get_config_var("LIBPL")
            if libpl and Path(libpl).is_dir():
                # Check common library names within LIBPL
                for name in potential_lib_names:
                    candidate = Path(libpl) / name
                    if candidate.exists():
                        python_library_file = str(candidate.resolve())
                        break

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_EXECUTABLE={python_executable}",
            f"-DPython_INCLUDE_DIR={python_include_dir}",
            # Only pass Python_LIBRARIES if we found the specific file
            f"-DPython_LIBRARIES={python_library_file}" if python_library_file else "",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-Dpybind11_FINDPYTHON=ON",
            "-DPython_FIND_STRATEGY=LOCATION",  # Prioritize using the executable location
        ]
        # Filter out empty strings from cmake_args (like potentially empty Python_LIBRARIES)
        cmake_args = [arg for arg in cmake_args if arg]

        # --- Rest of the CMake configuration and build steps remain the same ---

        # CMake generator and platform handling
        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        cmake_args += [f"-DTRIMCTS_VERSION_INFO={self.distribution.get_version()}"]

        build_args = ["--config", cfg]

        if self.compiler.compiler_type == "msvc":
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
            if not single_config:
                build_args += ["--", "/m"]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        # --- Build Execution ---
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)  # Ensure build temp exists

        print("-" * 10, "Running CMake prepare", "-" * 40)
        print(
            f"CMake command: cmake {ext.sourcedir} {' '.join(cmake_args)}"
        )  # Debug print
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True,
        )

        print("-" * 10, "Building extension", "-" * 43)
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True,
        )
        print("-" * 10, "Finished building extension", "-" * 36)

        # --- Copying (Fallback) ---
        # Check if the file was created in the expected location by CMake
        if not ext_fullpath.exists():
            module_name = ext.name.split(".")[-1]
            found = False
            # Search within the build temp directory
            for suffix in (".so", ".dylib", ".pyd"):
                candidates = list(build_temp.rglob(f"{module_name}*{suffix}"))
                if candidates:
                    built = candidates[0]
                    print(
                        f"Copying built extension from build temp: {built} -> {ext_fullpath}"
                    )
                    ext_fullpath.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(built, ext_fullpath)
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"Could not find built extension {module_name}.* in {extdir} or {build_temp}"
                )
        else:
            print(f"Found built extension at target: {ext_fullpath}")


class Develop(_develop):
    """Run CMake build_ext as part of 'python setup.py develop'."""

    def run(self):
        self.run_command("build_ext")
        super().run()


setup(
    # Metadata defined in pyproject.toml is preferred
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Include the py.typed file
    package_data={"trimcts": ["py.typed"]},
    ext_modules=[CMakeExtension("trimcts.trimcts_cpp", sourcedir="src/trimcts/cpp")],
    cmdclass={
        "build_ext": CMakeBuild,
        "develop": Develop,
    },
    zip_safe=False,
)
