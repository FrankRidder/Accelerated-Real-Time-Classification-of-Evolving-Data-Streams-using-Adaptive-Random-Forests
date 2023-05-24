#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "forest.hpp"

#include "hls_math.h"

static bool counterReset[TREE_COUNT][LEAFS_PER_TREE] = { false };
static bool nodeTypes[TREE_COUNT][NODE_COUNT_PER_TREE] = { false };
static node trees[TREE_COUNT][NODE_COUNT_PER_TREE] = { 0 };
static uint8_t outputClass[TREE_COUNT][NODE_COUNT_PER_TREE] = { 0 };

static uint16_t currentAmountOfLeafs[TREE_COUNT];

static bool forgroundTree[TREE_COUNT] = { false };
static bool treeActive[TREE_COUNT] = { false };

bool readActivity(const uint8_t treeIdx) {
#pragma HLS INLINE
	return treeActive[treeIdx];
}

bool readForground(const uint8_t treeIdx) {
#pragma HLS INLINE
	return forgroundTree[treeIdx];
}

node readForest(const uint8_t treeIdx, const uint16_t nodeIdx) {
#pragma HLS INLINE
#pragma HLS BIND_STORAGE variable=trees impl=uram type=ram_t2p
	return trees[treeIdx][nodeIdx];
}

bool readNodeType(const uint8_t treeIdx, const uint16_t nodeIdx) {
#pragma HLS INLINE
#pragma HLS BIND_STORAGE variable=nodeTypes impl=bram type=ram_t2p
	return nodeTypes[treeIdx][nodeIdx];
}

bool readNodeReset(const uint8_t treeIdx, const uint16_t nodeIdx) {
#pragma HLS INLINE
#pragma HLS BIND_STORAGE variable=nodeTypes impl=bram type=ram_t2p
	return counterReset[treeIdx][nodeIdx];
}

uint16_t readCurrentAmountOfLeaf(const uint8_t treeIdx) {
#pragma HLS INLINE
	return currentAmountOfLeafs[treeIdx];
}

uint8_t readOutputClassOfLeaf(const uint8_t treeIdx, const uint16_t nodeIdx) {
#pragma HLS INLINE
#pragma HLS BIND_STORAGE variable=outputClass impl=bram type=ram_t2p
	return outputClass[treeIdx][nodeIdx];
}

void updateTree(input_vector *inputVector,
		hls::stream<tree_info_vector> &treeInfoStream,
		hls::stream<train_output_vector> &trainOuput,
		hls::stream<adwin_output_vector> &adwinOutput,
		hls::stream<majority_class_output_vector> &majClassOuput,
		hls::stream<input_vector> &traverseInput) {

	static bool init = false;
	input_vector traverseInputVector = inputVector[0];

	update_tree_loop: for (uint8_t tree = 0; tree < TREES_PER_INSTANCE; tree++) {
#pragma HLS PIPELINE II=PIPELINE_II

		if (!init) {

			if (tree < FORGROUND_TREE_COUNT) {
				treeActive[tree] = true;
				forgroundTree[tree] = true;
				nodeTypes[tree][0] = true;
				currentAmountOfLeafs[tree] = 1;
				outputClass[tree][0] = traverseInputVector(ATTRIBUTE_COUNT_TOTAL + 7,
				ATTRIBUTE_COUNT_TOTAL);
			}

			if (tree == (TREE_COUNT - 1)) {
				init = true;
			}

			traverseInput.write(traverseInputVector);

		} else {
			traverseInput.write(traverseInputVector);
			train_output_vector trainingOutput = trainOuput.read();

			//Tree not active
			if (!trainingOutput.get_bit(25)) {
				continue;
			}

			uint8_t bestAttributeIndex = trainingOutput(7, 0);
			uint16_t newLeafIdx = trainingOutput(23, 8);
			bool split = trainingOutput.get_bit(24);

			tree_info_vector treeInfo = treeInfoStream.read();
			uint8_t treeIdx = treeInfo(7, 0);
			uint16_t currentFoundLeaf = treeInfo(23, 8);
			uint16_t leafNodeIdx = treeInfo(39, 24);

			adwin_output_vector AdwinOutputVector = adwinOutput.read();
			bool foundChange = AdwinOutputVector.get_bit(0);
			bool driftwarning = AdwinOutputVector.get_bit(1);

			majority_class_output_vector mjClassVector = majClassOuput.read();
			outputClass[treeIdx][currentFoundLeaf] = mjClassVector(7, 0);

			uint8_t backgroundIdx;

			//Make sure dectectorIdx is within range of FORGROUND_TREE_COUNT
			if (treeIdx >= FORGROUND_TREE_COUNT) {
				backgroundIdx = treeIdx - FORGROUND_TREE_COUNT;
			} else {
				backgroundIdx = treeIdx + FORGROUND_TREE_COUNT;
			}

			if (foundChange && !driftwarning) {
				//activate background tree
				treeActive[backgroundIdx] = true;
				nodeTypes[backgroundIdx][0] = true;
				currentAmountOfLeafs[backgroundIdx] = 1;
				counterReset[backgroundIdx][0] = true;
				outputClass[backgroundIdx][0] = mjClassVector(15, 8);
			} else if (foundChange && driftwarning) {
				//Replace tree
				forgroundTree[treeIdx] = false;
				forgroundTree[backgroundIdx] = true;
				treeActive[treeIdx] = false;
			} else {
				if (split) {
					trees[treeIdx][leafNodeIdx] = (uint16_t) bestAttributeIndex;
					nodeTypes[treeIdx][leafNodeIdx] = false;

					uint16_t leftNodeIdx = getLeft(leafNodeIdx);
					nodeTypes[treeIdx][leftNodeIdx] = true;
					trees[treeIdx][leftNodeIdx] = currentFoundLeaf;

					uint16_t rightNodeIdx = getRight(leafNodeIdx);
					nodeTypes[treeIdx][rightNodeIdx] = true;
					trees[treeIdx][rightNodeIdx] = newLeafIdx;

					counterReset[treeIdx][currentFoundLeaf] = true;
					counterReset[treeIdx][newLeafIdx] = true;

					outputClass[treeIdx][newLeafIdx] = mjClassVector(7, 0);
					currentAmountOfLeafs[treeIdx] = newLeafIdx + 1;
				} else {

					// If it needed to be reset it has been done now
					counterReset[treeIdx][currentFoundLeaf] = false;

				}
			}
		}
	}
}
