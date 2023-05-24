#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "hls_math.h"

#include "training_engine.hpp"
#include "forest.hpp"

#define SORT_DESCENDING 0
#define SORT_ASCENDING 1
#define DATA_TYPE float
#define KEY_TYPE uint8_t

uint8_t getAttribute(uint32_t &seed);
void computeInformationGain(const uint8_t treeIdx,
		const counter_type leafAttCounter[LEAF_COUNTER_ATT_SIZE],
		const counter_type leafAttClassTrueCounter[LEAF_COUNTER_ATT_CLASS_SIZE],
		const counter_type leafAttClassFalseCounter[LEAF_COUNTER_ATT_CLASS_SIZE],
		float &firstBestVal, float &secondBestVal, uint8_t &bestAttribute);
float computeHoeffdingBound(const uint32_t amountSeen);
void updateLeafCounters(const attribute_vector attributeVector,
		const uint8_t actualClass, const uint8_t weight,
		counter_type leafAttCounter[LEAF_COUNTER_ATT_SIZE],
		counter_type leafAttClassTrueCounter[LEAF_COUNTER_ATT_CLASS_SIZE],
		counter_type leafAttClassFalseCounter[LEAF_COUNTER_ATT_CLASS_SIZE]);

//update majority class of reached leaf
void updateMajorityClass(hls::stream<maj_count_input_vector> &majInputStream,
		counter_type *leafClassCounters,
		hls::stream<majority_class_output_vector> &majClassOuput) {
	tree_loop_maj: for (uint8_t tree = 0; tree < TREES_PER_INSTANCE; tree++) {
#pragma HLS PIPELINE II=PIPELINE_II

#pragma HLS DEPENDENCE dependent=false type=inter variable=leafClassCounters

		maj_count_input_vector majCountVector = majInputStream.read();

		//Tree inactive
		if (!majCountVector.get_bit(MAJ_INPUT_VECTOR_SIZE - 1)) {
			continue;
		}

		uint16_t currentFoundLeaf = majCountVector(15, 0);
		uint8_t correctClass = majCountVector(16 + 7, 16);
		uint8_t weight = majCountVector(16 + 8 + 7, 16 + 8);
		bool reset = majCountVector.get_bit(MAJ_INPUT_VECTOR_SIZE - 2);

		counter_type leafClassCounter[LEAF_COUNTER_CLASS_SIZE];
		uint32_t leafclassOffset = currentFoundLeaf * LEAF_COUNTER_CLASS_SIZE
				+ tree * LEAF_COUNTERS_CLASS_PER_TREE;

		if (reset) {
			counter_type zeros[LEAF_COUNTER_CLASS_SIZE] = { 0 };
			memcpy(leafClassCounter, zeros,
			LEAF_COUNTER_CLASS_SIZE * sizeof(counter_type));
		} else {
			// the counter start position corresponds to the leaf found of the current tree
			memcpy(leafClassCounter, leafClassCounters + leafclassOffset,
			LEAF_COUNTER_CLASS_SIZE * sizeof(counter_type));
		}

		leafClassCounter[correctClass] += weight;

		uint8_t majorityClassCode = 0;
		counter_type majorityClassCount = 0;

		update_majority_class: for (uint8_t classIdx = 0;
				classIdx < CLASS_COUNT; classIdx++) {
			counter_type count = leafClassCounter[classIdx];
			if (count > majorityClassCount) {
				majorityClassCount = count;
				majorityClassCode = classIdx;
			}
		}

		memcpy(leafClassCounters + leafclassOffset, leafClassCounter,
		LEAF_COUNTER_CLASS_SIZE * sizeof(counter_type));

		majority_class_output_vector mjClassVector = 0;
		mjClassVector(7, 0) = majorityClassCode;
		mjClassVector(15, 8) = correctClass;
		majClassOuput.write(mjClassVector);
	}
}

void train(hls::stream<train_input_vector1> &input1,
		hls::stream<train_input_vector2> &input2, counter_type *leafAttCounters,
		counter_type *leafAttClassTrueCounters,
		counter_type *leafAttClassFalseCounters,
		hls::stream<train_output_vector> &trainOutStream,
		hls::stream<tree_info_vector> &treeInfoOutputStream) {

	static counter_type leafSamplesSeenCountersArray[TREE_COUNT][LEAFS_PER_TREE] =
			{ 0 };
	static uint8_t leafGPCounterArray[TREE_COUNT][LEAFS_PER_TREE] = { 0 };

	tree_loop: for (uint8_t tree = 0; tree < TREES_PER_INSTANCE; tree++) {
#pragma HLS PIPELINE II=PIPELINE_II

#pragma HLS DEPENDENCE dependent=false type=inter variable=leafGPCounterArray
#pragma HLS DEPENDENCE dependent=false type=inter variable=leafAttCounters
#pragma HLS DEPENDENCE dependent=false type=inter variable=leafAttClassTrueCounters
#pragma HLS DEPENDENCE dependent=false type=inter variable=leafAttClassFalseCounters

		train_input_vector1 trainInputVector1 = input1.read();
		train_input_vector2 trainInputVector2 = input2.read();
		train_output_vector trainOutput = 0;

		//Tree inactive
		if (!trainInputVector1.get_bit(TRAINING_INPUT_VECTOR_SIZE - 1)) {
			trainOutStream.write(trainOutput);
			continue;
		}

		attribute_vector attributeVector = trainInputVector1(
				(ATTRIBUTE_COUNT_TOTAL - 1), 0);

		uint16_t currentFoundLeaf = trainInputVector2(15, 0);
		uint16_t leafNodeIdx = trainInputVector2(16 + 15, 16);
		uint8_t leafDepth = trainInputVector2(16 + 16 + 7, 16 + 16);
		uint8_t correctClass = trainInputVector2(16 + 16 + 8 + 7, 16 + 16 + 8);
		uint8_t weight = trainInputVector2(16 + 16 + 8 + 8 + 7,
				16 + 16 + 8 + 8);

		counter_type leafAttCounter[LEAF_COUNTER_ATT_SIZE];
		counter_type leafAttClassTrueCounter[LEAF_COUNTER_ATT_CLASS_SIZE];
		counter_type leafAttClassFalseCounter[LEAF_COUNTER_ATT_CLASS_SIZE];

#pragma HLS ARRAY_PARTITION variable=leafAttCounter type=complete
		DO_PRAGMA(HLS ARRAY_PARTITION variable=leafAttClassTrueCounter type=cyclic factor=ATTRIBUTE_COUNT_TOTAL)
		DO_PRAGMA(HLS ARRAY_PARTITION variable=leafAttClassFalseCounter type=cyclic factor=ATTRIBUTE_COUNT_TOTAL)

		uint32_t amountSeen =
				leafSamplesSeenCountersArray[tree][currentFoundLeaf] + 1;
		uint16_t currentAmountOfLeafs = readCurrentAmountOfLeaf(tree);
		uint8_t gracePeriod = leafGPCounterArray[tree][currentFoundLeaf];

		uint32_t leafAttOffset = currentFoundLeaf * LEAF_COUNTER_ATT_SIZE
				+ tree * LEAF_COUNTERS_ATT_PER_TREE;

		uint32_t leafStarterOffset = currentFoundLeaf
				* LEAF_COUNTER_ATT_CLASS_SIZE
				+ tree * LEAF_COUNTERS_ATT_CLASS_PER_TREE;

		bool reset = trainInputVector1.get_bit(TRAINING_INPUT_VECTOR_SIZE - 2);

		if (reset) {
			counter_type zeros[LEAF_COUNTER_ATT_CLASS_SIZE] = { 0 };

			amountSeen = 1;
			gracePeriod = 1;

			memcpy(leafAttCounter, zeros,
			LEAF_COUNTER_ATT_SIZE * sizeof(counter_type));
			memcpy(leafAttClassTrueCounter, zeros,
			LEAF_COUNTER_ATT_CLASS_SIZE * sizeof(counter_type));
			memcpy(leafAttClassFalseCounter, zeros,
			LEAF_COUNTER_ATT_CLASS_SIZE * sizeof(counter_type));
		} else {
			// the counter start position corresponds to the leaf found of the current tree
			memcpy(leafAttCounter, leafAttCounters + leafAttOffset,
			LEAF_COUNTER_ATT_SIZE * sizeof(counter_type));
			memcpy(leafAttClassTrueCounter,
					leafAttClassTrueCounters + leafStarterOffset,
					LEAF_COUNTER_ATT_CLASS_SIZE * sizeof(counter_type));
			memcpy(leafAttClassFalseCounter,
					leafAttClassFalseCounters + leafStarterOffset,
					LEAF_COUNTER_ATT_CLASS_SIZE * sizeof(counter_type));
		}

		updateLeafCounters(attributeVector, correctClass, weight,
				leafAttCounter, leafAttClassTrueCounter,
				leafAttClassFalseCounter);

		memcpy(leafAttCounters + leafAttOffset, leafAttCounter,
		LEAF_COUNTER_ATT_SIZE * sizeof(counter_type));
		memcpy(leafAttClassTrueCounters + leafStarterOffset,
				leafAttClassTrueCounter,
				LEAF_COUNTER_ATT_CLASS_SIZE * sizeof(counter_type));
		memcpy(leafAttClassFalseCounters + leafStarterOffset,
				leafAttClassFalseCounter,
				LEAF_COUNTER_ATT_CLASS_SIZE * sizeof(counter_type));

		leafSamplesSeenCountersArray[tree][currentFoundLeaf] = amountSeen;

		if ((gracePeriod == (GP - 1)) && (currentAmountOfLeafs < LEAFS_PER_TREE)
				&& (leafDepth < (MAX_TREE_DEPTH - 1))) {
			bool split;
			float firstBestVal;
			float secondBestVal;
			uint8_t bestAttribute;

			float hoeffdingBound = computeHoeffdingBound(amountSeen);

			computeInformationGain(tree, leafAttCounter,
					leafAttClassTrueCounter, leafAttClassFalseCounter,
					firstBestVal, secondBestVal, bestAttribute);

			if (((secondBestVal - firstBestVal) > hoeffdingBound)
					|| (hoeffdingBound < SPLIT_TIE_THRESHOLD)) {

				split = true;
			} else {
				split = false;
			}

			leafGPCounterArray[tree][currentFoundLeaf] = 0;

			trainOutput(7, 0) = bestAttribute;
			trainOutput(23, 8) = currentAmountOfLeafs;
			trainOutput.set_bit(24, split);
		} else {
			leafGPCounterArray[tree][currentFoundLeaf] = gracePeriod + 1;
		}

		tree_info_vector treeInfo;
		treeInfo(7, 0) = tree;
		treeInfo(23, 8) = currentFoundLeaf;
		treeInfo(39, 24) = leafNodeIdx;
		treeInfoOutputStream.write(treeInfo);

		trainOutput.set_bit(25, true);
		trainOutStream.write(trainOutput);

	}
}

// Update leaf counters with relevant information from new sample
// Also does online bagging by weighing the new information using Poisson distribution
void updateLeafCounters(const attribute_vector attributeVector,
		const uint8_t actualClass, const uint8_t weight,
		counter_type leafAttCounter[LEAF_COUNTER_ATT_SIZE],
		counter_type leafAttClassTrueCounter[LEAF_COUNTER_ATT_CLASS_SIZE],
		counter_type leafAttClassFalseCounter[LEAF_COUNTER_ATT_CLASS_SIZE]) {
#pragma HLS INLINE

	leafcounter_loop: for (uint8_t attributeIdx = 0;
			attributeIdx < ATTRIBUTE_COUNT_TOTAL; attributeIdx++) {
#pragma HLS UNROLL
		leafAttCounter[attributeIdx * AMOUNT_OF_POSSIBLE_VALUES
				+ attributeVector.get_bit(attributeIdx)] += weight;

		if (attributeVector.get_bit(attributeIdx)) {
			leafAttClassTrueCounter[attributeIdx
					+ actualClass * ATTRIBUTE_COUNT_TOTAL] += weight;
		} else {
			leafAttClassFalseCounter[attributeIdx
					+ actualClass * ATTRIBUTE_COUNT_TOTAL] += weight;
		}

	}
}

// Compute information gain
void computeInformationGain(const uint8_t treeIdx,
		const counter_type leafAttCounter[LEAF_COUNTER_ATT_SIZE],
		const counter_type leafAttClassTrueCounter[LEAF_COUNTER_ATT_CLASS_SIZE],
		const counter_type leafAttClassFalseCounter[LEAF_COUNTER_ATT_CLASS_SIZE],
		float &firstBestVal, float &secondBestVal, uint8_t &bestAttribute) {
#pragma HLS INLINE
	static bool init[TREE_COUNT] = { false };
	static uint32_t seeds[TREE_COUNT];
	uint32_t seed;

	if (!init[treeIdx]) {
		seed = SEED * (treeIdx + 1);
		init[treeIdx] = true;
	} else {
		seed = seeds[treeIdx];
	}

	float bestVal = 0.0f;
	float secBestVal = 0.0f;
	uint8_t bestAttributeo = 0;

	// Calculate the Gini impurity for every attribute for this leaf
	gini_attribute_loop: for (uint8_t attributeIdx = 0;
			attributeIdx < ATTRIBUTE_COUNT_PER_LEAF; attributeIdx++) {

		// get actual attribute index
		uint32_t attribute = getAttribute(seed);

		counter_type true_ij = leafAttCounter[(attribute
				* AMOUNT_OF_POSSIBLE_VALUES) + 1];
		counter_type false_ij = leafAttCounter[attribute
				* AMOUNT_OF_POSSIBLE_VALUES];

		float sumFalse = 0.0f;
		float sumTrue = 0.0f;

		gini_class_loop: for (uint32_t i = 0; i < CLASS_COUNT; i++) {

			float probaFalse = 0.0f;
			float probaTrue = 0.0f;

			float logProbaFalse = 0.0f;
			float logProbaTrue = 0.0f;

			counter_type true_ijk = leafAttClassTrueCounter[attribute
					+ i * ATTRIBUTE_COUNT_TOTAL];
			counter_type false_ijk = leafAttClassFalseCounter[attribute
					+ i * ATTRIBUTE_COUNT_TOTAL];

			// 0/0 = nan
			if (true_ij != 0.0f) {
				probaTrue = (float) true_ijk / true_ij;
			}
			if (false_ij != 0.0f) {
				probaFalse = (float) false_ijk / false_ij;
			}

			// log2(0) = -inf
			if (probaFalse != 0) {
				logProbaFalse = hls::log2f(probaFalse);
				sumFalse += probaFalse * -logProbaFalse;
			}
			if (probaTrue != 0) {
				logProbaTrue = hls::log2f(probaTrue);
				sumTrue += probaTrue * -logProbaTrue;
			}
		}

		float infoGain = sumFalse + sumTrue;

		if (attributeIdx == 0) {
			bestVal = infoGain;
			secBestVal = infoGain;
			bestAttributeo = attribute;
		} else if (infoGain < bestVal) {
			secBestVal = bestVal;
			bestVal = infoGain;
			bestAttributeo = attribute;
		} else if (infoGain < secBestVal || infoGain == bestVal) {
			secBestVal = infoGain;
		}
	}

	seeds[treeIdx] = seed;
	firstBestVal = bestVal;
	secondBestVal = secBestVal;
	bestAttribute = bestAttributeo;

}

// Select random set of attributes to try and split on
uint8_t getAttribute(uint32_t &value) {
#pragma HLS INLINE
	ap_uint<32> lfsr = value;

	// choose random set of attribute
	bool b_32 = lfsr.get_bit(0);
	bool b_22 = lfsr.get_bit(10);
	bool b_2 = lfsr.get_bit(30);
	bool b_1 = lfsr.get_bit(31);
	bool new_bit = b_32 ^ b_22 ^ b_2 ^ b_1;
	lfsr = lfsr >> 1;
	lfsr.set_bit(31, new_bit);

	value = lfsr;

	return (uint8_t) lfsr.to_uint() % ATTRIBUTE_COUNT_TOTAL;
}

// hoeffding bound
// providing an upper bound on the probability that the sum of a sample of independent random
// variables deviates from its expected value
//
// range: range of the random variable
// confidence / delta: desired probability of the estimate not being within the expected value
// n: the number of examples collected at the node
float computeHoeffdingBound(const uint32_t amountSeen) {
	// hoeffding bound parameters
#pragma HLS INLINE
	const float logConfidence = hls::logf((float) 1 / (float) TREE_DELTA);
	const float rangeHB = hls::log2f((float) CLASS_COUNT); // range of merit = log2(num_of_classes)
	const float hoeffdingInput = ((rangeHB * rangeHB) * logConfidence);

	return hls::sqrtf(hoeffdingInput / (2 * amountSeen));
}
