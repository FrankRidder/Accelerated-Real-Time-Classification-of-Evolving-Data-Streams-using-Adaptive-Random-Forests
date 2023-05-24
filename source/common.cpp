#include <stdio.h>
#include <stdint.h>

#include "common.hpp"

uint16_t getLeft(const uint16_t index) {
	return 2 * index + 1;
}

uint16_t getRight(const uint16_t index) {
	return 2 * index + 2;
}

