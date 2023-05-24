#ifndef TOP_LVL_H_
#define TOP_LVL_H_

#include "common.hpp"

extern "C" {
void top_lvl(input_vector *inputVector, counter_type *leafAttCounters,
		counter_type *leafClassCounters, counter_type *leafAttClassTrueCounters,
		counter_type *leafAttClassFalseCounters, uint8_t *output);
}
#endif /* TOP_LVL_H_ */
