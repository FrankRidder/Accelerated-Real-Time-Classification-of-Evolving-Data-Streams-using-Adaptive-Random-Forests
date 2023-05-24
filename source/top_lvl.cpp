#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "inference_engine.hpp"
#include "training_engine.hpp"
#include "adwin.hpp"
#include "forest.hpp"

#include "top_lvl.hpp"

extern "C" {

void top_lvl(input_vector *inputVector, counter_type *leafAttCounters, counter_type *leafClassCounters,
		counter_type *leafAttClassTrueCounters,
		counter_type *leafAttClassFalseCounters, uint8_t *output) {

#pragma HLS INTERFACE mode=s_axilite port=inputVector
	DO_PRAGMA(HLS interface mode=m_axi port=leafAttCounters depth=TOTAL_LEAF_COUNTERS_ATT bundle=BUS_1)
	DO_PRAGMA(HLS interface mode=m_axi port=leafClassCounters depth=TOTAL_LEAF_COUNTER_CLASS bundle=BUS_2)
	DO_PRAGMA(HLS interface mode=m_axi port=leafAttClassTrueCounters depth=TOTAL_LEAF_COUNTERS_ATT_CLASS bundle=BUS_3)
	DO_PRAGMA(HLS interface mode=m_axi port=leafAttClassFalseCounters depth=TOTAL_LEAF_COUNTERS_ATT_CLASS bundle=BUS_4)

#pragma HLS DATAFLOW

	//For more instances use split and merge
	hls::stream<input_vector> traverseInput("traverseInput");
	hls::stream<train_input_vector1> trainInput1("trainInput1");
	hls::stream<train_input_vector2> trainInput2("trainInput2");
	hls::stream<maj_count_input_vector> majClassInput("majInput");
	hls::stream<adwin_input_vector> adwinInput("adwinInput");
	hls::stream<vote_collection_input_vector> outputMerge("outputMerge");

	static hls::stream<adwin_output_vector, 20> adwinOutput(
			"adwinOutput");
	static hls::stream<train_output_vector, 20> trainOuput(
			"trainOuput");
	static hls::stream<majority_class_output_vector, 20> majClassOuput(
			"splittingOuput");
	static hls::stream<tree_info_vector, 20> treeInfoStream(
			"treeInfoStream");

	updateTree(inputVector, treeInfoStream, trainOuput, adwinOutput,
			majClassOuput, traverseInput);

	//Traverse
	traverse(traverseInput, trainInput1, trainInput2, adwinInput, majClassInput, outputMerge);

	train(trainInput1, trainInput2, leafAttCounters, leafAttClassTrueCounters, leafAttClassFalseCounters,
			trainOuput, treeInfoStream);

	updateMajorityClass(majClassInput, leafClassCounters, majClassOuput);

	updateAdwin(adwinInput, adwinOutput);

//Count collected votes
	countVotes(outputMerge, output);
}
}
