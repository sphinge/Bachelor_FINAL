# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wiktoria/Desktop/Thesis/NeuralNetwork

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wiktoria/Desktop/Thesis/NeuralNetwork

# Include any dependencies generated for this target.
include CMakeFiles/Neural.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Neural.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Neural.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Neural.dir/flags.make

CMakeFiles/Neural.dir/Neural.cpp.o: CMakeFiles/Neural.dir/flags.make
CMakeFiles/Neural.dir/Neural.cpp.o: Neural.cpp
CMakeFiles/Neural.dir/Neural.cpp.o: CMakeFiles/Neural.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wiktoria/Desktop/Thesis/NeuralNetwork/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Neural.dir/Neural.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Neural.dir/Neural.cpp.o -MF CMakeFiles/Neural.dir/Neural.cpp.o.d -o CMakeFiles/Neural.dir/Neural.cpp.o -c /home/wiktoria/Desktop/Thesis/NeuralNetwork/Neural.cpp

CMakeFiles/Neural.dir/Neural.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Neural.dir/Neural.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wiktoria/Desktop/Thesis/NeuralNetwork/Neural.cpp > CMakeFiles/Neural.dir/Neural.cpp.i

CMakeFiles/Neural.dir/Neural.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Neural.dir/Neural.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wiktoria/Desktop/Thesis/NeuralNetwork/Neural.cpp -o CMakeFiles/Neural.dir/Neural.cpp.s

# Object files for target Neural
Neural_OBJECTS = \
"CMakeFiles/Neural.dir/Neural.cpp.o"

# External object files for target Neural
Neural_EXTERNAL_OBJECTS =

Neural: CMakeFiles/Neural.dir/Neural.cpp.o
Neural: CMakeFiles/Neural.dir/build.make
Neural: /usr/lib/x86_64-linux-gnu/liblapack.so
Neural: /usr/lib/x86_64-linux-gnu/libcblas.so
Neural: /usr/lib/x86_64-linux-gnu/libatlas.so
Neural: /usr/local/lib/libshark.a
Neural: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
Neural: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
Neural: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
Neural: /usr/lib/x86_64-linux-gnu/liblapack.so
Neural: /usr/lib/x86_64-linux-gnu/libcblas.so
Neural: /usr/lib/x86_64-linux-gnu/libatlas.so
Neural: CMakeFiles/Neural.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wiktoria/Desktop/Thesis/NeuralNetwork/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Neural"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Neural.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Neural.dir/build: Neural
.PHONY : CMakeFiles/Neural.dir/build

CMakeFiles/Neural.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Neural.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Neural.dir/clean

CMakeFiles/Neural.dir/depend:
	cd /home/wiktoria/Desktop/Thesis/NeuralNetwork && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wiktoria/Desktop/Thesis/NeuralNetwork /home/wiktoria/Desktop/Thesis/NeuralNetwork /home/wiktoria/Desktop/Thesis/NeuralNetwork /home/wiktoria/Desktop/Thesis/NeuralNetwork /home/wiktoria/Desktop/Thesis/NeuralNetwork/CMakeFiles/Neural.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Neural.dir/depend

