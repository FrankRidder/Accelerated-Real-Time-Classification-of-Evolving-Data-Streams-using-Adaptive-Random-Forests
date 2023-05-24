#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>
#include <map>

//Add full path for synthesis
#include "top_lvl.hpp"


using namespace std;

uint16_t CLASS_COUNT_loc;
uint16_t ATTRIBUTE_COUNT_TOTAL_loc;
uint16_t ATTRIBUTE_COUNT_PER_TREE_loc;

void rm_nonprinting(std::string &str) {
	str.erase(std::remove_if(str.begin(), str.end(), [](unsigned char c) {
		return !std::isprint(c);
	}),
	str.end());
}

void write_csv(string filename, vector<float> data) {
	ofstream file(filename);

	if (!file.is_open()) {
		cerr << "Error: Failed to open " << filename << " for writing.\n";
		return;
	}

	for (double d : data) {
		file << d << "\n";
	}

	file.close();
}

vector<string> split(string str, string delim) {
	char *cstr = const_cast<char*>(str.c_str());
	char *current;
	vector<string> arr;
	current = strtok(cstr, delim.c_str());

	while (current != NULL) {
		arr.push_back(current);
		current = strtok(NULL, delim.c_str());
	}

	return arr;
}

bool prepare_data(ifstream &data_file, input_vector &input_data,
		map<string, int> &class_code_map) {

	string line;

	if (getline(data_file, line).fail()) {
		// reached end of line
		return false;
	}

	vector<string> raw_data_row = split(line, ",");

	for (uint8_t i = 0; i < ATTRIBUTE_COUNT_TOTAL; i++) {
		bool val = (bool) stoi(raw_data_row[i]);
		input_data.set(i, val);
	}

	//Catch linux end of line char
	rm_nonprinting(raw_data_row[ATTRIBUTE_COUNT_TOTAL]);

	input_data(ATTRIBUTE_COUNT_TOTAL + 7, ATTRIBUTE_COUNT_TOTAL) =
			class_code_map[raw_data_row[ATTRIBUTE_COUNT_TOTAL]];

	return true;
}

int main() {
	srand(time(NULL));

#ifdef COV // data path cov
	string data_path = "data/covtype/";
	string driftversion = "";
	string data_file_name = "covtype_binary_attributes.csv";
#endif

#ifdef KDD  //data path kddcup
	string data_path = "data/kddcup/";
	string driftversion = "";
	string data_file_name = "kddcup.csv";
#endif

#ifdef LED // data path led
	string data_path = "data/LED/";
	string driftversion = "led_2_w1/";

	//Random seed
	string seed = to_string(rand() % 10);
	std::cout << "seed = " << seed << std::endl;
	string data_file_name = seed.append(".csv");
#endif

#ifdef AGR 	// data path agr
	string data_path = "data/agrawal";
	string driftversion = "agrawal_2_w25k/";

	//Random seed
	string seed = to_string(rand() % 10);
	std::cout << "seed = " << seed << std::endl;
	string data_file_name = seed.append(".csv");
#endif

	// read data file
	string csv_path = data_path + driftversion + data_file_name;
	ifstream data_file(csv_path);

	// read attribute file
	string attribute_file_path = data_path + "/attributes.txt";
	ifstream attribute_file(attribute_file_path);

	// read class/label file
	string class_path = data_path + "/labels.txt";
	ifstream class_file(class_path);

	if (data_file) {
		std::cout << "Read data file" << std::endl;
	} else {
		std::cout << "Error reading data file" << std::endl;
		return 1;
	}

	if (attribute_file) {
		std::cout << "Read attribute file" << std::endl;
	} else {
		std::cout << "Error reading attribute file" << std::endl;
		return 1;
	}

	if (class_file) {
		std::cout << "Read class file" << std::endl;
	} else {
		std::cout << "Error reading class file" << std::endl;
		return 1;
	}

	string class_line;

	// init mapping between class and code
	map<string, int> class_code_map;
	map<int, string> code_class_map;

	vector<string> class_arr = split(class_line, " ");
	string code_str, class_str;

	int line_count = 0;
	while (class_file >> class_str) {
		int class_code = line_count;
		class_code_map[class_str] = class_code;
		code_class_map[class_code] = class_str;
		line_count++;
	}

	CLASS_COUNT_loc = line_count;

	std::cout << "CLASS_COUNT = " << CLASS_COUNT_loc << std::endl;

	// prepare attributes
	string line;
	getline(attribute_file, line);

	ATTRIBUTE_COUNT_TOTAL_loc = split(line, ",").size() - 1;
	ATTRIBUTE_COUNT_PER_TREE_loc = (int) sqrt(ATTRIBUTE_COUNT_TOTAL_loc) + 1;

	std::cout << "ATTRIBUTE_COUNT_TOTAL = " << ATTRIBUTE_COUNT_TOTAL_loc
			<< std::endl;
	std::cout << "ATTRIBUTE_COUNT_PER_TREE = " << ATTRIBUTE_COUNT_PER_TREE_loc
			<< std::endl;

	std::cout << "LEAF_COUNT_PER_TREE = " << (1 << (11 - 1)) << std::endl;
	std::cout << "NODE_COUNT_PER_TREE = " << ((1 << (11 - 1)) + (1 << (11 - 1)))
			<< std::endl;

	// Row 0 stores the total number of times value n_ij appeared.
	// Row 1 and onwards stores partial counters n_ijk for each class k.
	std::cout << "Allocating leaf counters " << std::endl;
	counter_type *leafAttCounters = (counter_type*) calloc(
	TOTAL_LEAF_COUNTERS_ATT, sizeof(counter_type));
	counter_type *leafClassCounters = (counter_type*) calloc(
	TOTAL_LEAF_COUNTER_CLASS, sizeof(counter_type));
	counter_type *leafAttClassTrueCounters = (counter_type*) calloc(
	TOTAL_LEAF_COUNTERS_ATT_CLASS, sizeof(counter_type));
	counter_type *leafAttClassFalseCounters = (counter_type*) calloc(
	TOTAL_LEAF_COUNTERS_ATT_CLASS, sizeof(counter_type));

	if (leafAttCounters == NULL || leafClassCounters == NULL
			|| leafAttClassTrueCounters == NULL
			|| leafAttClassFalseCounters == NULL) {
		std::cout << "Failed to allocate leaf counter space" << std::endl;
		return 1;
	}

	uint8_t predicted_class[1] = { 0 };
	uint32_t amountOfPredictions = 0;
	uint16_t batchCorrect = 0;
	uint16_t batchSize = 0;
	uint32_t amountOfCorrectPredictions = 0;
	input_vector input_data;
	input_vector input_array[1];
	vector<float> accuracy_data;

	float accuracy = 0;
	std::cout << "Starting testbench" << std::endl;
	while (true) {
		if (!prepare_data(data_file, input_data, class_code_map)) {
			break;
		}

		uint8_t actual_class = input_data(ATTRIBUTE_COUNT_TOTAL + 7,
		ATTRIBUTE_COUNT_TOTAL);

		input_array[0] = input_data;

		top_lvl(input_array, leafAttCounters, leafClassCounters,
				leafAttClassTrueCounters, leafAttClassFalseCounters,
				predicted_class);

		amountOfPredictions++;
		batchSize++;
		if (predicted_class[0] == actual_class) {
			amountOfCorrectPredictions++;
			batchCorrect++;
		}

		// Print accuracy every 1000 predictions
		if (batchSize == 1000) {
			accuracy = ((float) batchCorrect / batchSize) * 100;
			accuracy_data.push_back(accuracy);
			batchCorrect = 0;
			batchSize = 0;
			std::cout << "Accuracy = " << accuracy << std::endl;
		}
	}
	std::cout << std::endl;
	accuracy = ((float) amountOfCorrectPredictions / amountOfPredictions) * 100;
	std::cout << "Predictions = " << amountOfPredictions << std::endl;
	std::cout << "Correct predictions = " << amountOfCorrectPredictions
			<< std::endl;
	std::cout << "Accuracy = " << accuracy << std::endl;

	// Write accuracy data to CSV file
	write_csv(data_path.append(driftversion + "accuracy.csv"), accuracy_data);

	free(leafAttCounters);
	free(leafClassCounters);
	free(leafAttClassTrueCounters);
	free(leafAttClassFalseCounters);

	return 0;
}

