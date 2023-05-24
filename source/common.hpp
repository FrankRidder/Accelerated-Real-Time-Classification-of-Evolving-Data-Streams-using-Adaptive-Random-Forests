#ifndef COMMON_H_
#define COMMON_H_

#include <ap_int.h>
#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include <iostream>
#endif

//Change for dataset or make variable in a compile script
//Used for array definition
#define AGR
//#define COV
//#define KDD
//#define LED


#ifdef COV
#define CLASS_COUNT 						7
#define INCREASE_READ						(CLASS_COUNT + 1) //Used to align amount of 32 bits read with 512
#define ATTRIBUTE_COUNT_TOTAL 				44
#define ATTRIBUTE_COUNT_PER_LEAF 			7
#define TREE_COUNT 							200
#define PIPELINE_II							25
#define LOOP_PERF							(TREE_COUNT * PIPELINE_II)
#define LEAF_COUNTER_ATT_SIZE  				(ATTRIBUTE_COUNT_TOTAL * AMOUNT_OF_POSSIBLE_VALUES)
#define LEAF_COUNTER_CLASS_SIZE  			INCREASE_READ
#define LEAF_COUNTER_ATT_CLASS_SIZE 		(ATTRIBUTE_COUNT_TOTAL * INCREASE_READ)
#define AMOUNT_OF_INPUTS					581012
#endif


#ifdef KDD 
#define CLASS_COUNT 						23
#define INCREASE_READ						(CLASS_COUNT + 1) //Used to align amount of 32 bits read with 512
#define ATTRIBUTE_COUNT_TOTAL 				42
#define ATTRIBUTE_COUNT_PER_LEAF 			7
#define TREE_COUNT 							120
#define PIPELINE_II							63
#define LOOP_PERF							(TREE_COUNT * PIPELINE_II)
#define LEAF_COUNTER_ATT_SIZE  				(ATTRIBUTE_COUNT_TOTAL * AMOUNT_OF_POSSIBLE_VALUES)
#define LEAF_COUNTER_CLASS_SIZE  			CLASS_COUNT
#define LEAF_COUNTER_ATT_CLASS_SIZE 		(ATTRIBUTE_COUNT_TOTAL * INCREASE_READ)
#define AMOUNT_OF_INPUTS					494020
#endif


#ifdef LED 
#define CLASS_COUNT 						10
#define INCREASE_READ						CLASS_COUNT
#define ATTRIBUTE_COUNT_TOTAL 				24
#define ATTRIBUTE_COUNT_PER_LEAF 			5
#define TREE_COUNT 							120
#define PIPELINE_II							25
#define LOOP_PERF							(TREE_COUNT * PIPELINE_II)
#define LEAF_COUNTER_ATT_SIZE  				(ATTRIBUTE_COUNT_TOTAL * AMOUNT_OF_POSSIBLE_VALUES)
#define LEAF_COUNTER_CLASS_SIZE  			CLASS_COUNT
#define LEAF_COUNTER_ATT_CLASS_SIZE 	 	(ATTRIBUTE_COUNT_TOTAL * INCREASE_READ)
#define AMOUNT_OF_INPUTS					1000000
#endif


#ifdef AGR 
#define CLASS_COUNT 						2
#define ATTRIBUTE_COUNT_TOTAL 				82
#define INCREASE_READ						(ATTRIBUTE_COUNT_TOTAL + 6) //Used to align amount of 32 bits read with 512
#define ATTRIBUTE_COUNT_PER_LEAF 			10
#define TREE_COUNT 							200
#define PIPELINE_II							25
#define LOOP_PERF							(TREE_COUNT * PIPELINE_II)
#define LEAF_COUNTER_ATT_SIZE  				(INCREASE_READ * AMOUNT_OF_POSSIBLE_VALUES)
#define LEAF_COUNTER_CLASS_SIZE  			CLASS_COUNT
#define LEAF_COUNTER_ATT_CLASS_SIZE  		(INCREASE_READ * CLASS_COUNT)
#define AMOUNT_OF_INPUTS					1000000
#endif

//Offsets used for counter pointers
#define COUNTER_OFFSET 						1
#define AMOUNT_OF_POSSIBLE_VALUES 			2
#define LEAF_COUNTER_ROW_LENGTH 			(ATTRIBUTE_COUNT_TOTAL * AMOUNT_OF_POSSIBLE_VALUES)

//Tree input settings
#define AMOUNT_OF_INSTANCES					1
#define MAX_TREE_DEPTH  					11
#define FORGROUND_TREE_COUNT 				TREE_COUNT / 2
#define TREES_PER_INSTANCE					TREE_COUNT / AMOUNT_OF_INSTANCES


#define LEAFS_PER_TREE  					(1 << (MAX_TREE_DEPTH - 1))
#define NODE_COUNT_PER_TREE  				(LEAFS_PER_TREE + (1 << (MAX_TREE_DEPTH - 1)))
#define LEAF_COUNTERS_ATT_PER_TREE 			(LEAFS_PER_TREE * LEAF_COUNTER_ATT_SIZE)
#define LEAF_COUNTERS_CLASS_PER_TREE 		(LEAFS_PER_TREE * LEAF_COUNTER_CLASS_SIZE)
#define LEAF_COUNTERS_ATT_CLASS_PER_TREE 	(LEAFS_PER_TREE * LEAF_COUNTER_ATT_CLASS_SIZE)
#define TOTAL_LEAF_COUNTERS_ATT				(TREE_COUNT * LEAF_COUNTERS_ATT_PER_TREE)
#define TOTAL_LEAF_COUNTER_CLASS			(TREE_COUNT * LEAF_COUNTERS_CLASS_PER_TREE)
#define TOTAL_LEAF_COUNTERS_ATT_CLASS		(TREE_COUNT * LEAF_COUNTERS_ATT_CLASS_PER_TREE)
#define TOTAL_LEAF_COUNTERS					(TOTAL_LEAF_COUNTERS_ATT + TOTAL_LEAF_COUNTER_CLASS + TOTAL_LEAF_COUNTERS_ATT_CLASS * AMOUNT_OF_POSSIBLE_VALUES)

//ADWIN input
#ifdef KDD 
	#define WARNING_DELTA 					0.05
	#define DRIFT_DELTA 					0.005
#else
	#define WARNING_DELTA  					0.001
	#define DRIFT_DELTA 					0.00001
#endif

//Poisson input
#define LAMBDA 								1
#define SEED 								831250

//Training settings
#define TREE_DELTA 							0.05
#define GP 									16
#define SPLIT_TIE_THRESHOLD 				0.05


#define TRAINING_INPUT_VECTOR_SIZE 			(ATTRIBUTE_COUNT_TOTAL + 1 + 1)
#define MAJ_INPUT_VECTOR_SIZE 				(16 + 8 + 8 + 1 + 1)

// Usefull typedef for reduced magic numbers
typedef ap_uint<ATTRIBUTE_COUNT_TOTAL + 8> input_vector;
typedef ap_uint<ATTRIBUTE_COUNT_TOTAL> attribute_vector;
typedef ap_uint<16> node;
typedef ap_uint<FORGROUND_TREE_COUNT> output_vector;
typedef ap_uint<TREE_COUNT> status_vector;
typedef ap_uint<CLASS_COUNT> class_seen_vector;
typedef uint32_t counter_type;
typedef bool adwin_input;

typedef ap_uint<ATTRIBUTE_COUNT_TOTAL + 8> traverse_input_vector;
typedef ap_uint<TRAINING_INPUT_VECTOR_SIZE> train_input_vector1;
typedef ap_uint<8 + 16 + 16 + 8 + 8 > train_input_vector2;
typedef ap_uint<MAJ_INPUT_VECTOR_SIZE> maj_count_input_vector;
typedef ap_uint<8 + 16 + 1 + 1> train_output_vector;
typedef ap_uint<8 + 8> majority_class_output_vector;
typedef ap_uint<1 + 1 + 1> adwin_input_vector;
typedef ap_uint<1 + 1> adwin_output_vector;
typedef ap_uint<8 + 1 + 1> vote_collection_input_vector;
typedef ap_uint<64> leaf_info_vector;
typedef ap_uint<8 + 16> leaf_tree_vector;
typedef ap_uint<8 + 16 + 16> tree_info_vector;
typedef ap_uint<8> correct_class_vector;

typedef node tree[NODE_COUNT_PER_TREE];
typedef bool nodeTypeArray[NODE_COUNT_PER_TREE];
typedef uint8_t attributeArray[ATTRIBUTE_COUNT_PER_LEAF];
typedef float giniArray[ATTRIBUTE_COUNT_PER_LEAF];

//Define for checking bits
#define IS_BIT_SET(val, pos) (val & (1 << pos))

//Defines for adding variables in pragmas
#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

//Traverse nodess
uint16_t getLeft(const uint16_t index);
uint16_t getRight(const uint16_t index);

#endif
