#ifndef INFERENCE_ENGINE_H_
#define INFERENCE_ENGINE_H_

#include "common.hpp"

void countVotes(hls::stream<vote_collection_input_vector> &voteCollectionInput,
		uint8_t *output);
void traverse(hls::stream<input_vector> &traverseInput,
		hls::stream<train_input_vector1> &trainInput1,
		hls::stream<train_input_vector2> &trainInput2,
		hls::stream<adwin_input_vector> &adwinInput,
		hls::stream<maj_count_input_vector> &majInputStream,
		hls::stream<vote_collection_input_vector> &voteCollectionInput);

#endif /* INFERENCE_ENGINE_H_ */
