# Compiler
CC=g++
# Compiler flags
CFLAGS=-std=c++11 -Wall
# Libraries
LIBS=`pkg-config --cflags --libs opencv4`

# Source files
SRC=main.cpp

# Executable name
EXEC=my_program

# Make all
all: $(EXEC)

# Compile source files into object files
$(EXEC): $(SRC)
    $(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Clean
clean:
    rm -f $(EXEC)
