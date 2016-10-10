#ifndef neural_network_H
	#define neural_network_H
	#include <stdio.h>
	#include <stdbool.h>
	#include <stdint.h>
	//#include <stdbool.h>
	/* Mersenne twister variables */
	#define MT_N 624
	#define MT_M 397
	#define MT_UPPER_MASK 0x80000000UL
	#define MT_LOWER_MASK 0x7FFFFFFFUL
	#define RANDMAXV 2.3283064365387e-10
	#define PERTURBATION 1e-10
	
	#define LOCAL_MIN_EPOCHS 10
	
	enum nn_error_codes{
			NN_API_SUCCESS,	
			NN_API_NETWORK_ALLOC_ERROR,
			NN_API_EVALUATE_NETWORK,
			NN_API_RESCALE_DATA,
			NN_API_NOT_INIT,
			NN_API_BAD_INPUT,
			NN_API_WEIGHT_INIT,
			NN_API_DUPLICATE_NN,
			NN_API_COPY_DIFF,
			NN_API_COPY,
			NN_API_LS,
			NN_API_LS_SOLVE,
			NN_API_FREE,
			NN_API_TRAIN,
			NN_API_TRAIN2,
			NN_API_TRAIN3,
			NN_API_MALFORMED_NN,
			NN_API_GET_INV,
			NN_API_COMPUTE_JGH			
		};
	
	
	
	
	
	/* macro to protect function calls */
	#define C_SAFE_CALL(call)	do																					\
								{																					\
									call;																			\
									if( NN_API_ERROR != NN_API_SUCCESS )											\
									{																				\
										fprintf(stderr,"[ERROR] At function %s in line %d\n\twith message: %s\n",	\
													__FUNCTION__, __LINE__,	nn_error_strings[NN_API_ERROR] );		\
										NN_API_ERROR = 0;															\
										goto error;																	\
									}																				\
								}while(0)
	
	
								
	#define C_CHECK_CONDITION(cond, message)																					\
									do																							\
									{																							\
										if((cond) != 0)																			\
										{																						\
											fprintf(stderr, "[ERROR] At function %s (%s) in line %d \n\twith message: %s\n",	\
																	  __FUNCTION__, #cond,__LINE__,nn_error_strings[message]); 	\
											goto error;																			\
										}																						\
									}while(0)
								
	/* Function enumerators */
	enum net_func{ NN_LOGSIG, NN_TANSIG, NN_LINEAR, NN_NONE, net_func_nelem};
	  
	/******************************/
	/* Neural Network description */	
	/******************************/
	typedef struct net_layer{
    	   int32_t   num_inputs;              /* layer inputs                 */
    	   int32_t   num_outputs;             /* layer outputs                */
    	   enum      net_func eval_funct;     /* function to be used          */
    	   double ** weights;                 /* weights of this layer        */
    	   double *  bias;                    /* bias                         */
    	   double *  results;          		  /* layer results                */
	}net_layer_t;
	
	typedef struct net {
        
    	   /* network defition */
    	   int32_t num_layers;
        
    	   /* define input layer */
    	   double * in_layer;
        
    	   /* define output layer */
    	   double * out_layer;
        
    	   /* define network layers */
    	   net_layer_t ** layer;
		
		   /* input layer rescale parameters */
		   double * xmin;
		   double * xmax;
		   
		   /*  */
		   double * ymin;
		   double * ymax;
		   
		   /* net error code */
		   int32_t error_st;
		  
		   /* use mapping True by default (unless training) */
		   bool use_mapping;
		   
		   /* net init state */
		   bool init_state;
	}net_t;
 
	

	/* prototype defenitions */
	int 	dgelsd_ (int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *s, double *rcond, int *rank, double *work, int *lwork, int *iwork, int *info);
	int 	dgesv_ (int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
	void dgels_( char* trans, int* m, int* n, int* nrhs, double* a, int* lda, double* b, int* ldb, double* work, int* lwork, int* info );
	void dgetrf_(int *m,int *n, double * A, int *ncol, double *ipiv, int *info);
	void dgetri_(int *n,double * A, int * ncol, double *ipiv, double * work, int * lwork, int *info);
	net_t * create_nn(int32_t num_layers, int32_t * layers_conf, int32_t * layers_func );
	int32_t destroy_nn(net_t * nn);
	void * mem_alloc(int32_t size, bool zero_set);
	void mem_free(void * ptr);
	int32_t get_nn_response(net_t * nn);
	inline double map_min_max( double x, double xmin, double xmax, double ymin, double ymax, bool reverse );
	int32_t transform_in_out_data(net_t * nn, double * in_out, int32_t size, bool Mode);
	double compute_JGH(net_t * nn, double **J, double *G, double **H, double ** samples, double ** targets, double * errors, double * nn_out, double * nn_temp_out,
									 int32_t * perm, int32_t num_samples, int32_t num_outputs, int32_t G_length);
	int32_t nn_train(net_t * nn, double **samples, double ** targets, int32_t num_samples, int32_t max_epochs, double min_error );
	int32_t nn_train2(net_t * nn, double **samples, double ** targets, int32_t num_samples, int32_t max_epochs, double min_error, double train_perc );
	int32_t nn_train3(net_t * nn, double **samples, double ** targets, int32_t num_samples, int32_t max_epochs, double min_error);
	net_t * duplicate_network(net_t * nn );
	int32_t copy_nn_data(net_t *dest, net_t *src);
	void nn_grab_rescale_params(net_t * nn, double ** samples, double ** targets, int32_t num_samples );
	void nn_change_weights_bias(net_t * nn, double * G);
	int32_t init_nn_weights(net_t * nn);
	int32_t lpck_solve_ls(int32_t J_ydim, int32_t query, double * work, int32_t lwork, double * eq_matrix,  double * eq_sol);
	int32_t lpck_solve_nxn_ls(int32_t J_ydim, int32_t query, int32_t * work, int32_t lwork, double *eq_matrix, double * eq_sol);
	int32_t lpck_solve_ls_svd_pivot(int32_t J_ydim, int32_t query, double * work, int32_t lwork, double * S, int32_t * iwork,  double *eq_matrix, double * eq_sol);
	int32_t lpck_get_inv(int32_t J_ydim, double * work, int32_t lwork, double * ipiv, double * eq_matrix);
	void random_perm(int32_t num_p, int32_t * ranpat);
	void init_mt(void);
	unsigned long mt_random(void);
	
	
#endif	
