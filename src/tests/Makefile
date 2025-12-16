# Compiler settings
CXX = $(shell R CMD config CXX11)
CXXFLAGS = -std=c++11 -Wall -Wextra -I.. -I. $(shell Rscript -e "Rcpp:::CxxFlags()") $(shell Rscript -e "RcppArmadillo:::CxxFlags()")
# R libraries and linker flags (filtered to avoid problematic flags)
R_LIB = -L$(shell R RHOME)/lib -lR
LDFLAGS = $(R_LIB) $(shell R CMD config BLAS_LIBS) $(shell R CMD config LAPACK_LIBS)

# Source files to test
SOURCES_TO_TEST = ../utility.cpp ../Coxph.cpp

# Test files
TEST_SOURCES = test_main.cpp test_utility.cpp test_coxph.cpp test_data.cpp test_regression.cpp

# Object files
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)
LIB_OBJECTS = $(SOURCES_TO_TEST:.cpp=.o)

# Output executable
TARGET = test_runner

# Default target
all: $(TARGET)

# Build test runner
$(TARGET): $(TEST_OBJECTS) $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile test files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile library files
../%.o: ../%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run tests
test: $(TARGET)
	./$(TARGET)

# Run tests with verbose output
test-verbose: $(TARGET)
	./$(TARGET) -s

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TEST_OBJECTS) $(LIB_OBJECTS)

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Build test runner"
	@echo "  make test     - Build and run tests"
	@echo "  make test-verbose - Run tests with detailed output"
	@echo "  make clean    - Remove build artifacts"

.PHONY: all test test-verbose clean help
