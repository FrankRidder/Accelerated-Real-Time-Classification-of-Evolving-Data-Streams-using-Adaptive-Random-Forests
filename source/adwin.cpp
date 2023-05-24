#include <stdio.h>
#include <stdint.h>

#include "hls_math.h"

#include "adwin.hpp"

#define MAX_STORAGE_PER_BUCKETS 5
#define TOTAL_AMOUNT_OF_BUCKERS 16 //Aka W
#define MIN_CLOCK 32
//TODO try min of 10
#define MIN_SUBWINDOW_LENGTH 5

float getEstimation(const uint32_t TOTAL_CONTENT, const uint32_t TOTAL_WIDTH);
void compressBuckets(
		uint32_t buckets[TOTAL_AMOUNT_OF_BUCKERS][MAX_STORAGE_PER_BUCKETS],
		uint8_t currentIdx[TOTAL_AMOUNT_OF_BUCKERS],
		uint8_t currentSize[TOTAL_AMOUNT_OF_BUCKERS]);
bool setInput(const adwin_input input, const float delta,
		const uint8_t dectectorIdx, bool &adwinReset, uint32_t &TOTAL_WIDTH,
		uint32_t &TOTAL_CONTENT);
bool blnCutexpression(const uint32_t n0, const uint32_t n1, const float v1,
		const float mean, const float delta, const uint32_t width);

void updateAdwin(hls::stream<adwin_input_vector> &adwinInput,
		hls::stream<adwin_output_vector> &adwinOutput) {

#pragma HLS inline off

	tree_loop_adwin: for (uint8_t tree = 0; tree < TREES_PER_INSTANCE; tree++) {

		DO_PRAGMA(HLS performance target_ti=LOOP_PERF)
		static bool adwinReset[FORGROUND_TREE_COUNT] = { false };
		static uint32_t TOTAL_WIDTH[FORGROUND_TREE_COUNT] = { 0 };
		static uint32_t TOTAL_CONTENT[FORGROUND_TREE_COUNT] = { 0 };
		static bool driftwarning[FORGROUND_TREE_COUNT] = { false };

#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=TOTAL_WIDTH
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=TOTAL_CONTENT
#pragma HLS dependence variable=adwinReset type=inter false
#pragma HLS dependence variable=TOTAL_WIDTH type=inter false
#pragma HLS dependence variable=TOTAL_CONTENT type=inter false
#pragma HLS dependence variable=driftwarning type=inter false

		adwin_input_vector adwinInputVector = adwinInput.read();
		adwin_output_vector output = 0;

		//Tree inactive
		if (!adwinInputVector.get_bit(1)) {
			continue;
		}

		//Tree not a forground tree so no detector
		if (!adwinInputVector.get_bit(2)) {
			adwinOutput.write(output);
			continue;
		}

		uint8_t dectectorIdx;
		float delta;

		bool input = adwinInputVector.get_bit(0);

		//Make sure dectectorIdx is within range of FORGROUND_TREE_COUNT
		if (tree >= FORGROUND_TREE_COUNT) {
			dectectorIdx = tree - FORGROUND_TREE_COUNT;
		} else {
			dectectorIdx = tree;
		}

		float old_error = getEstimation(TOTAL_CONTENT[dectectorIdx],
				TOTAL_WIDTH[dectectorIdx]);

		if (!driftwarning[dectectorIdx]) {
			delta = WARNING_DELTA;
		} else {
			delta = DRIFT_DELTA;
		}

		bool foundChange = setInput(input, delta, dectectorIdx,
				adwinReset[dectectorIdx], TOTAL_WIDTH[dectectorIdx],
				TOTAL_CONTENT[dectectorIdx]);

		//If error decreasing keep the tree going
		if (foundChange
				&& old_error
						> getEstimation(TOTAL_CONTENT[dectectorIdx],
								TOTAL_WIDTH[dectectorIdx])) {
			foundChange = false;
		}

		output.set_bit(0, foundChange);
		output.set_bit(1, driftwarning[dectectorIdx]);

		adwinOutput.write(output);

		if (foundChange && !driftwarning[dectectorIdx]) {
			driftwarning[dectectorIdx] = true;
		} else if (foundChange && driftwarning[dectectorIdx]) {
			driftwarning[dectectorIdx] = false;
			adwinReset[dectectorIdx] = true;
		}
	}
}

bool setInput(const adwin_input input, const float delta,
		const uint8_t dectectorIdx, bool &adwinReset, uint32_t &TOTAL_WIDTH,
		uint32_t &TOTAL_CONTENT) {
	static uint32_t buckets[FORGROUND_TREE_COUNT][TOTAL_AMOUNT_OF_BUCKERS][MAX_STORAGE_PER_BUCKETS] =
			{ 0 };
	static uint8_t currentIdx[FORGROUND_TREE_COUNT][TOTAL_AMOUNT_OF_BUCKERS] = {
			0 };
	static uint8_t currentSize[FORGROUND_TREE_COUNT][TOTAL_AMOUNT_OF_BUCKERS] =
			{ 0 };
	static float VARIANCE[FORGROUND_TREE_COUNT] = { 0.0f };
	static uint8_t mintTime[FORGROUND_TREE_COUNT] = { 0 };

#pragma HLS INLINE

#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=buckets
#pragma HLS BIND_STORAGE variable=buckets type=ram_t2p
#pragma HLS ARRAY_PARTITION dim=1 type=cyclic factor=4 variable=currentIdx
#pragma HLS ARRAY_PARTITION dim=1 type=cyclic factor=4 variable=currentSize
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=VARIANCE

#pragma HLS dependence variable=buckets type=inter false
#pragma HLS dependence variable=currentIdx type=inter false
#pragma HLS dependence variable=currentSize type=inter false
#pragma HLS dependence variable=adwinReset type=inter false
#pragma HLS dependence variable=mintTime type=inter false
#pragma HLS dependence variable=VARIANCE type=inter false
#pragma HLS dependence variable=TOTAL_CONTENT type=inter false
#pragma HLS dependence variable=TOTAL_WIDTH type=inter false

	bool blnChange = false;
	bool blnExit = false;

	if (adwinReset) {
		//1)Increment window in one element
		buckets[dectectorIdx][0][0] = input;
		TOTAL_CONTENT = input;
		TOTAL_WIDTH = 1;
		VARIANCE[dectectorIdx] = 0;
		mintTime[dectectorIdx] = 1;

		currentIdx[dectectorIdx][0] = 1;
		currentSize[dectectorIdx][0] = 1;

		adwin_reset_loop: for (uint8_t bucket = 1;
				bucket < TOTAL_AMOUNT_OF_BUCKERS; ++bucket) {
			currentIdx[dectectorIdx][bucket] = 0;
			currentSize[dectectorIdx][bucket] = 0;
		}
		adwinReset = false;

	} else {
		//1)Increment window in one element
		buckets[dectectorIdx][0][currentIdx[dectectorIdx][0]] = input;

		TOTAL_WIDTH++;

		if (TOTAL_WIDTH > 1) {
			VARIANCE[dectectorIdx] += (float) (TOTAL_WIDTH - 1)
					* ((float) input
							- (float) TOTAL_CONTENT / (float) (TOTAL_WIDTH - 1))
					* ((float) input
							- (float) TOTAL_CONTENT / (float) (TOTAL_WIDTH - 1))
					/ (float) TOTAL_WIDTH;

		}

		if (input) {
			TOTAL_CONTENT++;
		}

		//virtual shifting
		if (currentIdx[dectectorIdx][0] == (MAX_STORAGE_PER_BUCKETS - 1)) {
			currentIdx[dectectorIdx][0] = 0;
		} else {
			currentIdx[dectectorIdx][0]++;
		}

		//2) Compress Buckets
		compressBuckets(buckets[dectectorIdx], currentIdx[dectectorIdx],
				currentSize[dectectorIdx]);

		//3)Reduce  window
		if (mintTime[dectectorIdx] == (MIN_CLOCK - 1)) {
			mintTime[dectectorIdx] = 0;

			uint8_t bucketIdx;
			uint32_t n0 = 0;
			uint32_t n1 = TOTAL_WIDTH;
			uint32_t u0 = 0;
			uint32_t u1 = TOTAL_CONTENT;

			adwin_bucket_loop: for (int8_t i = (TOTAL_AMOUNT_OF_BUCKERS - 1);
					i >= 0; i--) {

				if (blnExit)
					break;
				//Bucket empty
				if (currentSize[dectectorIdx][i] == 0)
					continue;

				//Get current bucket Idx
				if (currentIdx[dectectorIdx][i] - currentSize[dectectorIdx][i]
						>= 0) {
					bucketIdx = currentIdx[dectectorIdx][i]
							- currentSize[dectectorIdx][i];
				} else {
					int8_t offset = currentIdx[dectectorIdx][i]
							- currentSize[dectectorIdx][i];
					bucketIdx = MAX_STORAGE_PER_BUCKETS + offset;
				}

				adwin_subbucket_loop: for (uint8_t k = 0;
						k < MAX_STORAGE_PER_BUCKETS; k++) {
					if (k == currentSize[dectectorIdx][i])
						break;

					uint32_t bucketSize = hls::exp2((uint32_t) i);

					n0 += bucketSize;
					n1 -= bucketSize;
					u0 += buckets[dectectorIdx][i][bucketIdx];
					u1 -= buckets[dectectorIdx][i][bucketIdx];

					float mean = ((float) u0 / n0) - ((float) u1 / n1);

					if ((n1 >= MIN_SUBWINDOW_LENGTH
							&& n0 >= MIN_SUBWINDOW_LENGTH)
							&& blnCutexpression(n0, n1, VARIANCE[dectectorIdx],
									mean, delta, TOTAL_WIDTH)) {
						blnChange = true;
					}

					//Keep idx within bounds
					bucketIdx++;

					if (bucketIdx == MAX_STORAGE_PER_BUCKETS)
						bucketIdx = 0;

				} //Next subbucket
			} //Next Bucket
		} else {
			mintTime[dectectorIdx]++;
		}
	}
	return blnChange;
}

void compressBuckets(
		uint32_t buckets[TOTAL_AMOUNT_OF_BUCKERS][MAX_STORAGE_PER_BUCKETS],
		uint8_t currentIdx[TOTAL_AMOUNT_OF_BUCKERS],
		uint8_t currentSize[TOTAL_AMOUNT_OF_BUCKERS]) {

#pragma HLS dependence variable=buckets type=inter false
#pragma HLS dependence variable=currentIdx type=inter false
#pragma HLS dependence variable=currentSize type=inter false

#pragma HLS INLINE

	uint16_t additionValue1;
	uint16_t additionValue2;

	uint8_t sizeOfCurrentBucket = currentSize[0];

	if (sizeOfCurrentBucket == (MAX_STORAGE_PER_BUCKETS - 1)) {
		uint8_t idxOfCurrentBucket = currentIdx[0];
		uint8_t idxOfNextBucket = currentIdx[1];
		uint8_t sizeOfNextBucket = currentSize[1];
		//Check if array index for addition is within bound
		if (idxOfCurrentBucket == (MAX_STORAGE_PER_BUCKETS - 1)) {
			additionValue1 = buckets[0][(MAX_STORAGE_PER_BUCKETS - 1)];
			additionValue2 = buckets[0][0];
		} else {
			additionValue1 = buckets[0][idxOfCurrentBucket];
			additionValue2 = buckets[0][idxOfCurrentBucket + 1];
		}

		//2) Compress buckets
		adwin_compress_loop: for (uint8_t k = 0;
				k < TOTAL_AMOUNT_OF_BUCKERS - 1; k++) {
			buckets[k + 1][idxOfNextBucket] = (additionValue1 + additionValue2);
			//virtual shifting
			if (idxOfNextBucket == (MAX_STORAGE_PER_BUCKETS - 1)) {
				idxOfNextBucket = 0;
			} else {
				idxOfNextBucket++;
			}

			currentSize[k] = sizeOfCurrentBucket - 1;

			if (sizeOfNextBucket != (MAX_STORAGE_PER_BUCKETS - 1)) {

				currentSize[k + 1] = sizeOfNextBucket + 1;
				currentIdx[k + 1] = idxOfNextBucket;

				break;
			} else {
				if (idxOfNextBucket == (MAX_STORAGE_PER_BUCKETS - 1)) {
					additionValue1 =
							buckets[k + 1][(MAX_STORAGE_PER_BUCKETS - 1)];
					additionValue2 = buckets[k + 1][0];
				} else {
					additionValue1 = buckets[k + 1][idxOfNextBucket];
					additionValue2 = buckets[k + 1][idxOfNextBucket + 1];
				}

				idxOfCurrentBucket = idxOfNextBucket;
				currentIdx[k + 1] = idxOfNextBucket;
				idxOfNextBucket = currentIdx[k + 2];

				sizeOfCurrentBucket = sizeOfNextBucket;
				sizeOfNextBucket = currentSize[k + 2];
			}
		}
	} else {
		currentSize[0]++;

	}
}

bool blnCutexpression(const uint32_t n0, const uint32_t n1, const float v,
		const float mean, const float delta, const uint32_t width) {

	float variance = v / width;

	float dd = hls::logf(2 * hls::logf((float) width) / delta); // -- ull perque el ln n va al numerador. Formula Gener 2008

	float m = ((float) 1 / (n0 - MIN_SUBWINDOW_LENGTH + 1))
			+ ((float) 1 / (n1 - MIN_SUBWINDOW_LENGTH + 1));

	float epsilon = hls::sqrtf(2 * m * variance * dd) + (float) 2 / 3 * dd * m;

	return (hls::fabs(mean) > epsilon);
}

float getEstimation(const uint32_t TOTAL_CONTENT, const uint32_t TOTAL_WIDTH) {
	return ((float) TOTAL_CONTENT / TOTAL_WIDTH);

}
