#ifndef FOREST_H_
#define FOREST_H_

#include "common.hpp"

bool readActivity(const uint8_t treeIdx);
bool readForground(const uint8_t treeIdx);
uint8_t readOutputClassOfLeaf(const uint8_t treeIdx, const uint16_t nodeIdx);
uint16_t readCurrentAmountOfLeaf(const uint8_t treeIdx);
bool readNodeType(const uint8_t treeIdx, const uint16_t nodeIdx);
node readForest(const uint8_t treeIdx, const uint16_t nodeIdx);
bool readNodeReset(const uint8_t treeIdx, const uint16_t nodeIdx);
void updateTree(input_vector* inputVector,
		hls::stream<tree_info_vector> &treeInfoStream,
		hls::stream<train_output_vector> &trainOuput,
		hls::stream<adwin_output_vector> &adwinOutput,
		hls::stream<majority_class_output_vector> &majClassOuput,
		hls::stream<input_vector> &traverseInput);

#endif /* FOREST_H_ */
