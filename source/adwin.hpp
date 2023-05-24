#ifndef ADWIN_H_
#define ADWIN_H_

#include "common.hpp"

void updateAdwin(hls::stream<adwin_input_vector> &adwinInput,
		hls::stream<adwin_output_vector> &adwinOutput);

#endif /* ADWIN_H_ */
