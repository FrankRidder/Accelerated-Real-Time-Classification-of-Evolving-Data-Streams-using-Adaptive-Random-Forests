#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "hls_math.h"

#include "inference_engine.hpp"
#include "forest.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

uint8_t poisson(const uint8_t treeIdx);
ap_uint<3> popcount5Bits(const ap_uint<5> x);
ap_uint<7> popcount100Bits(const output_vector x);

void traverse(hls::stream<input_vector> &traverseInput,
		hls::stream<train_input_vector1> &trainInput1,
		hls::stream<train_input_vector2> &trainInput2,
		hls::stream<adwin_input_vector> &adwinInput,
		hls::stream<maj_count_input_vector> &majInputStream,
		hls::stream<vote_collection_input_vector> &voteCollectionInput) {

#pragma HLS inline off

	tree_loop: for (uint8_t tree = 0; tree < TREES_PER_INSTANCE; tree++) {
#pragma HLS PIPELINE II=PIPELINE_II
		input_vector travInput = traverseInput.read();

		adwin_input_vector adwinInputVector = 0;
		train_input_vector1 trainingInput1 = 0;
		train_input_vector2 trainingInput2 = 0;
		vote_collection_input_vector voteInput = 0;
		maj_count_input_vector majCountInput = 0;

		bool treeActive = readActivity(tree);

		if (!treeActive) {
			adwinInput.write(adwinInputVector);
			voteCollectionInput.write(voteInput);
			trainInput1.write(trainingInput1);
			trainInput2.write(trainingInput2);
			majInputStream.write(majCountInput);
			continue;
		}

		attribute_vector attributeVector = travInput(
		ATTRIBUTE_COUNT_TOTAL - 1, 0);

		bool found = false;
		uint16_t nodeIdx = 0;
		uint16_t leafIdx = 0;
		uint8_t depthLeaf;

		traverse_loop: for (uint8_t depth = 0; depth < MAX_TREE_DEPTH;
				depth++) {
			if (!found) {
				node treeNode = readForest(tree, nodeIdx);
				//If node equals leaf
				if (readNodeType(tree, nodeIdx) == true) {
					leafIdx = treeNode;
					depthLeaf = depth;
					found = true;
				} else {
					uint8_t nodeClass = (uint8_t) treeNode(7, 0);
					nodeIdx =
							attributeVector.get_bit(nodeClass) == false ?
									getLeft(nodeIdx) : getRight(nodeIdx);

				}
			}
		}

		uint8_t actualClass = travInput(ATTRIBUTE_COUNT_TOTAL + 7,
		ATTRIBUTE_COUNT_TOTAL);

		uint8_t predictedClass = readOutputClassOfLeaf(tree, leafIdx);
		bool forground = readForground(tree);
		uint8_t weight = poisson(tree);

		adwinInputVector.set_bit(0,
				predictedClass == actualClass ? false : true);
		adwinInputVector.set_bit(1, true);
		adwinInputVector.set_bit(2, forground);
		adwinInput.write(adwinInputVector);

		bool reset = readNodeReset(tree, leafIdx);
		trainingInput1((ATTRIBUTE_COUNT_TOTAL - 1), 0) = attributeVector;
		trainingInput1.set_bit((TRAINING_INPUT_VECTOR_SIZE - 2), reset);
		trainingInput1.set_bit((TRAINING_INPUT_VECTOR_SIZE - 1), true);
		trainInput1.write(trainingInput1);

		trainingInput2(15, 0) = leafIdx;
		trainingInput2(16 + 15, 16) = nodeIdx;
		trainingInput2(16 + 16 + 7, 16 + 16) = depthLeaf;
		trainingInput2(16 + 16 + 8 + 7, 16 + 16 + 8) = actualClass;
		trainingInput2(16 + 16 + 8 + 8 + 7, 16 + 16 + 8 + 8) = weight;
		trainInput2.write(trainingInput2);


		majCountInput(15, 0) = leafIdx;
		majCountInput(16 + 7, 16) = actualClass;
		majCountInput(16 + 8 + 7, 16 + 8) = weight;
		majCountInput.set_bit((MAJ_INPUT_VECTOR_SIZE - 2), reset);
		majCountInput.set_bit((MAJ_INPUT_VECTOR_SIZE - 1), true);
		majInputStream.write(majCountInput);

		voteInput(7, 0) = predictedClass;
		voteInput.set_bit(8, true);
		voteInput.set_bit(9, forground);
		voteCollectionInput.write(voteInput);
	}
}

// Vote counting to use when multiple trees write at the same time
void countVotesOld(hls::stream<vote_collection_input_vector> &voteCollectionInput,
		uint8_t *output) {
#pragma HLS inline off
	uint8_t outputCount = 0;
	uint8_t majorityClass = 0;
	uint8_t majorityClassCount = 0;

	/**
	 * 2d array
	 * Rows -> classes
	 * columns -> trees
	 * Bit set if tree votes class
	 */
	output_vector collectedVotes[CLASS_COUNT] = { 0 };

#pragma HLS ARRAY_PARTITION variable=collectedVotes type=complete

	uint8_t voteIdx = 0;

	input_read_loop: for (uint8_t tree = 0; tree < TREE_COUNT; tree++) {
#pragma HLS PIPELINE II=PIPELINE_II
		vote_collection_input_vector voteCollectionInputVector =
				voteCollectionInput.read();

		//Tree is not a forground tree or not a active tree
		if (voteCollectionInputVector.get_bit(8)
				&& voteCollectionInputVector.get_bit(9)) {
			uint8_t outputClass = voteCollectionInputVector(7, 0);
			collectedVotes[outputClass].set_bit(voteIdx, true);
			voteIdx++;
		}

		if (tree == (TREE_COUNT - 1)) {
			DO_PRAGMA(HLS occurrence cycle=TREE_COUNT)
			popcount_loop: for (uint8_t classCount = 0;
					classCount < CLASS_COUNT; classCount++) {

				outputCount = popcount100Bits(collectedVotes[classCount]);

				if (majorityClassCount < outputCount) {
					majorityClassCount = outputCount;
					majorityClass = classCount;
				}
			}
			output[0] = majorityClass;
		}
	}
}

void countVotes(hls::stream<vote_collection_input_vector> &voteCollectionInput,
		uint8_t *output) {
#pragma HLS inline off
	uint8_t majorityClass = 0;
	uint8_t majorityClassCount = 0;

	/**
	 * 2d array
	 * Rows -> classes
	 * columns -> trees
	 * Bit set if tree votes class
	 */
	uint8_t collectedVotes[CLASS_COUNT] = { 0 };

#pragma HLS ARRAY_PARTITION variable=collectedVotes type=complete

	input_read_loop: for (uint8_t tree = 0; tree < TREE_COUNT; tree++) {
#pragma HLS PIPELINE II=PIPELINE_II
		vote_collection_input_vector voteCollectionInputVector =
				voteCollectionInput.read();

		//Tree is not a forground tree or not a active tree
		if (voteCollectionInputVector.get_bit(8)
				&& voteCollectionInputVector.get_bit(9)) {
			uint8_t outputClass = voteCollectionInputVector(7, 0);
			collectedVotes[outputClass]++;
		}

		if (tree == (TREE_COUNT - 1)) {
			DO_PRAGMA(HLS occurrence cycle=TREE_COUNT)
			popcount_loop: for (uint8_t classCount = 0;
					classCount < CLASS_COUNT; classCount++) {

				if (majorityClassCount < collectedVotes[classCount]) {
					majorityClassCount = collectedVotes[classCount];
					majorityClass = classCount;
				}
			}
			output[0] = majorityClass;
		}
	}
}

//Should become a LUT
ap_uint<3> popcount5Bits(const ap_uint<5> x) { //pop count for 5 bits
	switch (x) {
	case 0:
		return 0;
	case 1:
		return 1;
	case 2:
		return 1;
	case 3:
		return 2;
	case 4:
		return 1;
	case 5:
		return 2;
	case 6:
		return 2;
	case 7:
		return 3;
	case 8:
		return 1;
	case 9:
		return 2;
	case 10:
		return 2;
	case 11:
		return 3;
	case 12:
		return 2;
	case 13:
		return 3;
	case 14:
		return 3;
	case 15:
		return 4;
	case 16:
		return 1;
	case 17:
		return 2;
	case 18:
		return 2;
	case 19:
		return 3;
	case 20:
		return 2;
	case 21:
		return 3;
	case 22:
		return 3;
	case 23:
		return 4;
	case 24:
		return 2;
	case 25:
		return 3;
	case 26:
		return 3;
	case 27:
		return 4;
	case 28:
		return 3;
	case 29:
		return 4;
	case 30:
		return 4;
	case 31:
		return 5;
	}
	return 6; // illegal for a 5 bits value input!
}

// Use pipeline (Done is example )
ap_uint<7> popcount100Bits(const output_vector x) {
	ap_uint<7> output = 0;
	bigpopcount_loop: for (uint8_t bit = 0; bit < FORGROUND_TREE_COUNT; bit +=
			5) {
		output += popcount5Bits(x(bit + 4, bit));
	}
	return output;
}

uint8_t poisson(const uint8_t treeIdx) {
	static bool init[TREE_COUNT] = { false };
	static uint32_t seeds[TREE_COUNT];
	ap_uint<10> rand;
	ap_uint<32> lfsr;
	float product = 1.0f;
	float sum = 1.0f;

	if (!init[treeIdx]) {
		lfsr  = SEED * (treeIdx + 1);
		init[treeIdx] = true;
	} else {
		lfsr = seeds[treeIdx];
	}

	// Poisson distribution using LFSR
	// Change lambda in header file
	const uint8_t max_val = (10 * LAMBDA); //max(100, 10 * (int) (lambda))
	const float expLambda = hls::expf(LAMBDA);

	random_bit_loop: for (uint8_t currentBit = 0; currentBit < 10;
			currentBit++) {

		bool b_32 = lfsr.get_bit(0);
		bool b_22 = lfsr.get_bit(10);
		bool b_2 = lfsr.get_bit(30);
		bool b_1 = lfsr.get_bit(31);
		bool new_bit = b_32 ^ b_22 ^ b_2 ^ b_1;
		lfsr = lfsr >> 1;
		lfsr.set_bit(31, new_bit);

		rand.set(currentBit, new_bit);
	}

	seeds[treeIdx] = lfsr;

	float next_double = (float) rand / 1023;
	float threshold = next_double * expLambda;

	uint8_t i;
	poisson_loop: for (i = 1; i < max_val; i++) {
		if (sum > threshold)
			break;
		product *= ((float) LAMBDA / i);
		sum += product;
	}

	return i - 1;
}
