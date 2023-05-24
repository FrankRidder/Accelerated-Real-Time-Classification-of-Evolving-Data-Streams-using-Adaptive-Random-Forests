#ifndef TRAINING_ENGINE_H_
#define TRAINING_ENGINE_H_

#include "common.hpp"

void train(hls::stream<train_input_vector1> &input1,
		hls::stream<train_input_vector2> &input2,
		counter_type *leafAttCounters,
		counter_type *leafAttClassTrueCounters,
		counter_type *leafAttClassFalseCounters,
		hls::stream<train_output_vector> &trainOutStream,
		hls::stream<tree_info_vector> &treeInfoOutputStream);
void updateMajorityClass(
		hls::stream<maj_count_input_vector> &majInputStream,
		counter_type *leafClassCounters,
		hls::stream<majority_class_output_vector> &majClassOuput);

#endif /* TRAINING_ENGINE_H_ */
