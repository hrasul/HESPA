# configuration
TEST_CASE = julia
BUILD_DIR = ../../build/$(TEST_CASE)

# default parameters
WIDTH    = 1920
HEIGHT   = 1080
MAX_ITER = 128
C_REAL   = -0.8
C_IMAG   = 0.2

PARAMETERS = $(WIDTH) $(HEIGHT) $(MAX_ITER) $(C_REAL) $(C_IMAG)

targets = $(BUILD_DIR)/$(TEST_CASE)

.PHONY: all
all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(TEST_CASE): $(TEST_CASE).cu lodepng.cpp lodepng.h
	nvc++ -O3 -o $(BUILD_DIR)/$(TEST_CASE) $(TEST_CASE).cu lodepng.cpp
.PHONY: $(TEST_CASE)
$(TEST_CASE): $(BUILD_DIR)/$(TEST_CASE)

.PHONY: run
run: all
	cd $(BUILD_DIR) && ./$(TEST_CASE) $(PARAMETERS)

.PHONY: clean
clean:
	rm -f $(targets)