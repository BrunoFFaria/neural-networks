/* File description : Implementation of neural networks
 * Author : Bruno Faria
 * Year : 2014 
 * Work place : University of Aveiro
*/
#include <time.h>
#include <stdlib.h>
#include <stdio.h> 
#include <string.h>
#include <math.h>
#include "neural_network.h"


//#include "mkl.h"
//#include <atlas_enum.h>
//#include <lapacke.h>
//#include <lapacke.h>
const char * nn_error_strings[19] = { "API succeed",
								  	  "API could not allocate memory for network object",
								  	  "API could not evaluate network object",
								  	  "API could not rescale data",
								  	  "neural network not initialized",
								  	  "API bad input",
								  	  "API weight initialization",
 								  	  "API could not duplicate network",
								  	  "API could not copy data from src to dest, networks are different",
								  	  "API could not copy data :s",
								  	  "API could not solve system of equations (lapack least squares)",
								  	  "API lapack error",
								  	  "API could not free memory (what?)",
								  	  "API train",
								  	  "API train 2",
								  	  "API train 3",
								  	  "API could not duplicate network",
								  	  "API could not get inverse",
								  	  "API could not compute J, G, and H"
								  };

#define log2(x) log(x)/0.30102999566
/* Random number generator vars */
unsigned long mt_state[MT_N];
unsigned long mt_constants[2];
int mt_state_idx;
bool rand_state = false;

/* global stack error */
int32_t NN_API_ERROR = 0;

/* =======================================
 * = Allocate and destroy neural network =
 * =======================================
*/
net_t * create_nn(int32_t num_layers, int32_t * layers_conf, int32_t * layers_func )
{
   int32_t i = 0, j = 0;
   net_t * nn = NULL;
   
   C_CHECK_CONDITION( num_layers == 0 || num_layers > 200 || num_layers < 1, NN_API_BAD_INPUT);
  
   /* allocate a neural network object */
   C_SAFE_CALL( nn = (net_t *)mem_alloc(sizeof(net_t), true ) );
   

   /* allocate the layers field pointers */
   C_SAFE_CALL( nn->layer = (net_layer_t **)mem_alloc(num_layers * sizeof(net_layer_t *), true) );
   
   /* set num layers */
   nn -> num_layers = num_layers;
   nn -> init_state = false;
   
   /* allocate layers vector */
   C_SAFE_CALL( nn->layer[0] = (net_layer_t *)mem_alloc(num_layers * sizeof(net_layer_t), true) );
   
   /* generate layers pointers */
   for(i = 1; i < num_layers; i++) { nn->layer[i] = nn->layer[0] + i; }
   
   /* allocate space for the layers */
   for(i = 0; i < num_layers; i++)
   {
   		
   		C_CHECK_CONDITION( layers_conf[i] <= 0 || layers_conf[i] > 400, NN_API_BAD_INPUT );
   		
   		
   		/* insert number of inputs and outputs of the layer */
   		if(i == 0)
   		{
   			C_CHECK_CONDITION( layers_func[i] !=  NN_NONE  || layers_func[i] > net_func_nelem, NN_API_BAD_INPUT );
   			/* input layer */
   			nn->layer[i]->num_inputs = 0;
   			nn->layer[i]->num_outputs = layers_conf[i]; 
            nn->layer[i]->eval_funct = NN_NONE;
            nn->layer[i]->weights = NULL;
            nn->layer[i]->bias = NULL;
			C_SAFE_CALL( nn->layer[i]->results = (double*)mem_alloc( layers_conf[i] * sizeof(double), true ));  

		    /* allocate memory for input and output mapping */
   			C_SAFE_CALL( nn->xmin = (double *)mem_alloc(layers_conf[i] * sizeof(double),true ));
   			C_SAFE_CALL( nn->xmax = (double *)mem_alloc(layers_conf[i] * sizeof(double),true ));
   			 
   		}else{
   			
   			C_CHECK_CONDITION( layers_func[i] ==  NN_NONE  || layers_func[i] > net_func_nelem, NN_API_BAD_INPUT );
			
   			/* allocate memory for input and output mapping */
   			C_SAFE_CALL( nn->ymin = (double *)mem_alloc(layers_conf[i] * sizeof(double),true ));
   			C_SAFE_CALL( nn->ymax = (double *)mem_alloc(layers_conf[i] * sizeof(double),true ));
			
   			/* remaining layers */
   			nn->layer[i]->num_inputs = layers_conf[i-1];
   			nn->layer[i]->num_outputs = layers_conf[i];
        	nn->layer[i]->eval_funct = layers_func[i];
                
            C_SAFE_CALL( nn->layer[i]->weights 	 	 = (double**)mem_alloc( layers_conf[i] 				  	* sizeof(double 	*), true ) );
            C_SAFE_CALL( nn->layer[i]->results 	 	 = (double *)mem_alloc( layers_conf[i] 					* sizeof(double		 ), true ) );
            C_SAFE_CALL( nn->layer[i]->bias 		 = (double *)mem_alloc( layers_conf[i] 					* sizeof(double		 ), true ) );
            C_SAFE_CALL( nn->layer[i]->weights[0] 	 = (double *)mem_alloc( layers_conf[i-1] * layers_conf[i] * sizeof(double		 ), true ) );
            
            for(j = 1; j < layers_conf[i]; j++)
            {
            	nn->layer[i]->weights[j] = nn->layer[i]->weights[0] + j * layers_conf[i-1];
            }
       }
   }
   nn -> init_state = true;
   nn -> use_mapping = true;
   return nn;

error:
    /* something went wrong so clean memory and exit */
    destroy_nn( nn );
        
    return NULL;
}

int32_t destroy_nn(net_t * nn)
{
    int32_t i = 0;
    net_layer_t * layer;
  
    
    C_CHECK_CONDITION( nn == NULL, NN_API_BAD_INPUT );
    C_CHECK_CONDITION( nn->num_layers == 0, NN_API_BAD_INPUT );

    /* first deallocate the layers */
    if(nn->layer != NULL)
    {
    	if(nn->layer[0] != NULL)
    	{
			layer = nn->layer[0];
			mem_free(layer->results);
    		for(i = 1; i < nn->num_layers; i++ )
    		{
    			layer = nn->layer[i];
    			if(layer->weights != NULL)
    			{
    				mem_free( layer->weights[0] );
        			mem_free( layer->weights 	);
        		}
        		mem_free( layer->results 	 );
       	 		mem_free( layer->bias    	 );
    		}
    		/* now free the main object */
   			mem_free( nn->layer[0] );
   		}
   		mem_free( nn->layer );
    }
    
    mem_free( nn->xmin );
    mem_free( nn->xmax );
    mem_free( nn->ymin );
    mem_free( nn->ymax );
    
    mem_free( nn );
   
    return 0;
error:
    /* what? cannot deallocate memory? why? */
    return -1;
}
 
void * mem_alloc(int32_t size, bool zero_set)
{
     void * p = malloc(size);
    
     if( p == NULL){ NN_API_ERROR = NN_API_NETWORK_ALLOC_ERROR; return NULL;}
     
     if(zero_set == true)
     {
        memset(p, 0, size);
     }
     NN_API_ERROR = NN_API_SUCCESS;
     return p;
}

void mem_free(void * ptr)
{
	C_CHECK_CONDITION(ptr == NULL, NN_API_BAD_INPUT);
    free( ptr );
    ptr = NULL;
    NN_API_ERROR = NN_API_SUCCESS;
	return;
error:
	NN_API_ERROR = NN_API_FREE;
	return;
}


/* get the neural network response for a given input */
int32_t get_nn_response(net_t * nn)
{
	int32_t i = 0, n = 0, l = 0;
	double w_sum = 0.0f;

	net_layer_t * layer = NULL, * prev_layer = NULL;

	C_CHECK_CONDITION( nn == NULL, 				  NN_API_BAD_INPUT 		);
	C_CHECK_CONDITION( nn -> init_state == false, NN_API_BAD_INPUT 		);
	C_CHECK_CONDITION( nn -> layer == NULL,		  NN_API_MALFORMED_NN	);
	
	memcpy(nn->layer[0]->results, nn->in_layer,nn->layer[0]->num_outputs * sizeof(double));
	
	/* first layer hols the data to be evaluated */
	if(nn->use_mapping){
		C_SAFE_CALL( transform_in_out_data(nn, nn->layer[0]->results, nn->layer[0]->num_outputs, false) );
	}
	
	for(l = 1; l < nn->num_layers; l++)
	{
		C_CHECK_CONDITION( nn -> layer[l] 	== NULL,	NN_API_MALFORMED_NN	);
		C_CHECK_CONDITION( nn -> layer[l-1] == NULL,	NN_API_MALFORMED_NN	);
		
		/* remap layer (to increase readibility) */
		layer = nn->layer[l]; 
		prev_layer = nn->layer[l-1];
		
		C_CHECK_CONDITION( layer -> bias 		== NULL,	NN_API_MALFORMED_NN	);
		C_CHECK_CONDITION( layer -> weights 	== NULL,	NN_API_MALFORMED_NN	);
		C_CHECK_CONDITION( layer -> weights[0] 	== NULL,	NN_API_MALFORMED_NN	);
		
		for(n = 0; n < layer->num_outputs; n++)
		{
			
			w_sum = layer->bias[n];
			
			for( i = 0; i < layer->num_inputs; i++)
				w_sum += prev_layer->results[i] * layer->weights[n][i];
			
			// apply layer function
			switch( layer->eval_funct )
			{
				case NN_LOGSIG: { w_sum =  1 / (1 + exp(-w_sum)); 			break; }
				case NN_TANSIG: { w_sum =  2 / (1 + exp(-2 * w_sum)) - 1;	break; }
				default: 		{ 											break; }
			}
			
			layer->results[n] = w_sum;
		}
	}

	/* transform the data and output it to the outer layer */
	layer = nn->layer[nn->num_layers-1];
	memcpy(nn->out_layer, layer->results, layer->num_outputs * sizeof(double));
	if(nn->use_mapping)
	{
		C_SAFE_CALL( transform_in_out_data(nn, nn->out_layer, layer->num_outputs, true) );
	}
	
	NN_API_ERROR = NN_API_SUCCESS;
	return 0;
error:
	NN_API_ERROR = NN_API_RESCALE_DATA;
	return -1;
}


inline double map_min_max( double x, double xmin, double xmax, double ymin, double ymax, bool reverse )
{
	if(reverse)
	{
		return (ymax-ymin)*(x+1.0f)/2.0f + ymin;
	}else{
		return 2.0f * (x - xmin)/(xmax-xmin) - 1.0f;
	}
}

int32_t transform_in_out_data(net_t * nn, double * in_out, int32_t size, bool Mode)
{
	int32_t i = 0;

	C_CHECK_CONDITION( size == 0, NN_API_BAD_INPUT);
	
	for(i = 0; i < size; i++){
		if(Mode){
			in_out[i] = map_min_max(in_out[i], 0, 0, nn->ymin[i], nn->ymax[i], Mode);
		}else{
			in_out[i] = map_min_max(in_out[i], nn->xmin[i], nn->xmax[i], 0, 0, Mode);
		}
	}
	NN_API_ERROR = NN_API_SUCCESS;
	return 0;
	
error:
	NN_API_ERROR = NN_API_RESCALE_DATA;
	return -1;
}

/*
 * Compute stuff
 */
 
double compute_JGH(net_t * nn, double **J, double *G, double **H, double ** samples, double ** targets, double * errors, double * nn_out, double * nn_temp_out,
 int32_t * perm, int32_t num_samples, int32_t num_outputs, int32_t G_length)
{
	double sse = 0;
	int32_t i = 0, j = 0, k = 0, l = 0, s = 0, ct = 0;
	net_layer_t * layer = NULL;
	double weight_s = 0;
	
	/* verify conditions */
	C_CHECK_CONDITION( nn 			== NULL, 				  NN_API_BAD_INPUT 	);
	C_CHECK_CONDITION( J  			== NULL, 				  NN_API_BAD_INPUT 	);
	C_CHECK_CONDITION( G  			== NULL,				  NN_API_BAD_INPUT	);
	C_CHECK_CONDITION( samples  	== NULL, 				  NN_API_BAD_INPUT 	);
	C_CHECK_CONDITION( targets  	== NULL,				  NN_API_BAD_INPUT	);
	C_CHECK_CONDITION( errors  		== NULL, 				  NN_API_BAD_INPUT 	);
	C_CHECK_CONDITION( nn_out  		== NULL,				  NN_API_BAD_INPUT	);
	C_CHECK_CONDITION( nn_temp_out  == NULL, 				  NN_API_BAD_INPUT 	);
	C_CHECK_CONDITION( num_samples  == 0, 					  NN_API_BAD_INPUT 	);
	C_CHECK_CONDITION( num_outputs  == 0,					  NN_API_BAD_INPUT	);
	C_CHECK_CONDITION( G_length  	== 0,					  NN_API_BAD_INPUT	);
	
	/* compute sample errors and jacobian */
	for(s = 0; s < num_samples; s++)
	{
		ct = 0;
		
		/* short circuit the pointers */
		nn->in_layer = samples[perm[s]]; nn->out_layer = nn_out;
		
		/* first evaluate the network for this sample without any perturbation*/
		C_SAFE_CALL( get_nn_response( nn ) );
		
		/* compute errors and sse */
		for(i = 0; i < num_outputs; i++)
		{
			errors[i*num_samples + s] = targets[perm[s]][i] - nn_out[i];
			sse += errors[i*num_samples + s]*errors[i*num_samples + s];
		}
		
		nn->out_layer = nn_temp_out;
		
		/* the lines of the jacobian for this sample are computed here */
		for(l = 1; l < nn->num_layers; l++)
		{
			layer = nn->layer[l];
		
			for(i = 0; i < layer->num_outputs; i++)
			{	
				/* weights perturbation */
				for(j = 0; j < layer->num_inputs; j++)
				{
			
					weight_s = layer->weights[i][j];
					
					/* apply perturbation parameter */
					layer->weights[i][j] += (PERTURBATION * ( 1 + abs(weight_s)));
				
					/* get neural network response */
					C_SAFE_CALL( get_nn_response( nn ) );
				
					/* restore weight */
					layer->weights[i][j] = weight_s;
				
					/* compute derivative */
					for(k = 0; k < num_outputs; k++){
						J[k*num_samples+s][ct] = (nn_out[k] - nn_temp_out[k]) / (PERTURBATION * ( 1 + abs(weight_s)));
					}
					ct++;	
				}
			
				/* bias perturbation */
				weight_s = layer->bias[i];
				
				/* apply perturbation parameter */
				layer->bias[i] += (PERTURBATION * ( 1 + abs(weight_s)));
				
				/* get neural network response */
				C_SAFE_CALL( get_nn_response( nn ) );
				
				/* restore weight */
				layer->bias[i] = weight_s;
				
				/* compute derivative */
				for(k = 0; k < num_outputs; k++)
					J[k*num_samples+s][ct] = (nn_out[k] - nn_temp_out[k]) / (PERTURBATION * ( 1 + abs(weight_s)));
				ct++;
			}
		
		}
	}
	
	/* compute gradient and aproximation to the hessian matrix */
	memset(G, 0, G_length * sizeof(double));
	memset(H[0], 0, G_length * G_length * sizeof(double));
		
	for(i = 0; i < num_samples * num_outputs; i++ )
	{
		for(j = 0; j < G_length; j++)
		{
			G[j] += J[i][j] * errors[i];
		}
	}
		
	/* symmetric matrix ??? Do we have singularities? */
	for(i = 0; i < G_length; i++)
	{
		for(j = 0; j < G_length; j++)
		{
			for(k = 0; k < num_samples * num_outputs; k++)
				H[i][j] += J[k][i]*J[k][j];
		}
	}
	NN_API_ERROR = NN_API_SUCCESS;;
	return sse;
error:
	NN_API_ERROR = NN_API_COMPUTE_JGH;
	return 0;
}
/* ===================================== 
 * = Neural network training functions =
 * =====================================
 */
int32_t nn_train(net_t * nn, double **samples, double ** targets, int32_t num_samples, int32_t max_epochs, double min_error )
{
	double ** J = NULL, ** A = NULL, * A_temp = NULL, * G = NULL, * errors = NULL, * nn_out = NULL, * nn_t_out = NULL, * work = NULL, * S = NULL;
	double niu = 0.01, current_error = 1e20, new_error = 0;
	double **r_samples = NULL, **r_targets = NULL;
	int32_t * iwork = NULL; 
	int32_t l = 0, J_ydim = 0, nlvl = 0, epoch = 0, *perm = NULL, lwork = 0, s = 0, num_outputs = 0, num_inputs = 0;
	int32_t i = 0;
	net_layer_t *layer = NULL;
	net_t * test_nn = NULL;
	
	/* memory for rescaled samples */
	
	/* conditions check */
	C_CHECK_CONDITION( nn == NULL, 							 NN_API_NOT_INIT  );
	C_CHECK_CONDITION( nn->init_state == false, 			 NN_API_NOT_INIT  );
	C_CHECK_CONDITION( num_samples <= 0, 					 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( max_epochs < 0 && max_epochs > 10000, NN_API_BAD_INPUT );
	C_CHECK_CONDITION( min_error < 0 && min_error > 1000,	 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( samples == NULL, 					 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( targets == NULL, 					 NN_API_BAD_INPUT );
	
	/* init weights */
	C_SAFE_CALL( init_nn_weights(nn) );
	
	/* first and last layer : required to know the number of inputs and outputs and so allocate memory */
	layer = nn->layer[nn->num_layers-1];
	num_outputs = layer->num_outputs;
	
	layer = nn->layer[0];
	num_inputs = layer->num_outputs;
	
	/* compute jacobian number of lines */
	for(l = 1; l < nn->num_layers; l++)
	{
		layer = nn->layer[l];
		J_ydim += (layer->num_inputs + 1) * layer-> num_outputs;
	}
	printf("J_ydim %d\n",J_ydim);
	
	
	/* initialize required memory */
	C_SAFE_CALL( J 		  = (double **)mem_alloc(num_outputs * num_samples	* sizeof(double  *), true) );
	C_SAFE_CALL( A 		  = (double **)mem_alloc(J_ydim     				* sizeof(double  *), true) );
	C_SAFE_CALL( G 	 	  = (double  *)mem_alloc(J_ydim		  				* sizeof(double   ), true) );
	C_SAFE_CALL( errors   = (double  *)mem_alloc(num_outputs * num_samples 	* sizeof(double   ), true) );
	C_SAFE_CALL( perm     = (int32_t *)mem_alloc(num_samples 				* sizeof(int32_t  ), true) );
	C_SAFE_CALL( nn_out   = (double  *)mem_alloc(num_outputs		   		* sizeof(double   ), true) );
	C_SAFE_CALL( nn_t_out = (double  *)mem_alloc(num_outputs		   		* sizeof(double   ), true) );
	
	C_SAFE_CALL( J[0]     	  = (double  *)mem_alloc(num_outputs * num_samples *  J_ydim * sizeof(double ), true) );
	C_SAFE_CALL( A[0]      	  = (double  *)mem_alloc(J_ydim * J_ydim  				 	 * sizeof(double ), true) );
	C_SAFE_CALL( A_temp   	  = (double  *)mem_alloc(J_ydim * J_ydim  					 * sizeof(double ), true) );
	C_SAFE_CALL( r_samples    = (double **)mem_alloc(num_samples 						 * sizeof(double*), true) );
	C_SAFE_CALL( r_samples[0] = (double  *)mem_alloc(num_samples * num_inputs 			 * sizeof(double*), true) );
	C_SAFE_CALL( r_targets    = (double **)mem_alloc(num_samples 						 * sizeof(double*), true) );
	C_SAFE_CALL( r_targets[0] = (double  *)mem_alloc(num_samples * num_outputs 			 * sizeof(double*), true) );
	
	/* calculate pointers address */
	for(l = 1; l < J_ydim; l++)
		A[l] = A[0] + l * J_ydim; 
		
	for(l = 1; l < num_outputs * num_samples; l++)
		J[l] = J[0] + l * J_ydim;
	
	for(l=1; l < num_samples; l++){
		r_samples[l] = r_samples[0] + l * num_inputs;
		r_targets[l] = r_targets[0] + l * num_outputs;
	}
	
	
	/* query lapack for the best size of the work area */
	nlvl = (int32_t)(floor(log2((double)J_ydim/(25.0f+1.0f)))+1.0f);

	C_SAFE_CALL( S 		= (double  *) mem_alloc( J_ydim 							 * sizeof(double ), true) );
	C_SAFE_CALL( iwork 	= (int32_t *) mem_alloc( ( 3 * J_ydim * nlvl + 11 * J_ydim ) * sizeof(int32_t), true) );  
	lwork = lpck_solve_ls_svd_pivot(J_ydim, -1, NULL, 0, S, iwork, A[0], G);
	C_SAFE_CALL( work = (double *)mem_alloc(lwork * sizeof(double), true) );
	
	/* grab scalling parameters */
	nn_grab_rescale_params( nn, samples, targets, num_samples );
	
	/* rescale samples and targets to -1, 1 range */
	for(l = 0; l < num_samples; l++){
		memcpy(r_samples[l], samples[l], num_inputs * sizeof(double));
		C_SAFE_CALL( transform_in_out_data(nn, r_samples[l], num_inputs, false) );

		memcpy(r_targets[l], targets[l], num_outputs * sizeof(double));
		for(i = 0; i < num_outputs; i++){
				r_targets[l][i] = map_min_max(r_targets[l][i], nn->ymin[i], nn->ymax[i], 0, 0, false);
		}
		
	}
	
	/* generate a duplicate network */
	C_SAFE_CALL( test_nn = duplicate_network( nn ) );
	
	/* disable rescaling */
	nn->use_mapping = false;
	
	/* generate random presentation */
	random_perm(num_samples, perm);
	C_SAFE_CALL( copy_nn_data( test_nn, nn ) );
	
	/* compute jacobian, hessian and stuff */
	current_error = compute_JGH(test_nn, J, G, A, r_samples, r_targets, errors, nn_out, nn_t_out, perm, num_samples, num_outputs, J_ydim);
	
	/* lets train the neural network */
	while(sqrt(current_error) > min_error && epoch < max_epochs && niu < 1e20)
	{
		
		/* copy network to a safe place */
		C_SAFE_CALL( copy_nn_data( test_nn, nn ) );
		
		memcpy(A_temp, A[0], J_ydim * J_ydim * sizeof(double));
		
		/* apply niu */
		for(i = 0; i < J_ydim; i++)
			 A_temp[i*J_ydim+i] += niu ;
		
		/* call lapack to solve our system of equations */
		C_SAFE_CALL( lpck_solve_ls_svd_pivot(J_ydim, 0, work, lwork, S, iwork, A_temp, G) );
		
		/* apply changes to neural network weights (lapack gives solution in G) */
		nn_change_weights_bias(test_nn, G);
		
		/* compute new error */
		new_error = 0.0f;
		for(s = 0; s < num_samples; s++)
		{
			/* short circuit the pointers */
			test_nn->in_layer = r_samples[perm[s]]; test_nn->out_layer = nn_out;
			
			/* compute neural network response for this input */
			C_SAFE_CALL( get_nn_response( test_nn ) );
			
			/* compute sample errors */
			for(i = 0; i < num_outputs; i++)
				new_error += (r_targets[perm[s]][i] - nn_out[i]) * (r_targets[perm[s]][i] - nn_out[i]);
		}
		
		/* check wether or not we should accept this network */
		if(new_error < current_error )
		{
			/* copy network */
			C_SAFE_CALL( copy_nn_data( nn, test_nn ) );
			current_error = new_error;
			niu *= 0.1;
			random_perm(num_samples, perm);
			
			/* compute new hessian, jacobian and gradient */
			current_error = compute_JGH(test_nn, J, G, A, r_samples, r_targets, errors, nn_out, nn_t_out, perm, num_samples, num_outputs, J_ydim);
	
		}else{
			niu *= 10;
		}
		
		printf("epoch: %d error : %f niu : %f\n",epoch,sqrt(current_error),niu);
		
		epoch++;
	}
	
	/* clean memory and exit */
	destroy_nn( test_nn );
	
	/* enable rescaling */
	nn->use_mapping = true;
	
	if(r_samples != NULL)
	{
		mem_free(r_samples[0]);
		mem_free(r_samples);
	}

	if(r_targets != NULL)
	{
		mem_free(r_targets[0]);
		mem_free(r_targets);
	}

	if(J != NULL)
	{
		mem_free( J[0] 	);
		mem_free( J 	);
	}
	if(A != NULL)
	{
		mem_free( A[0] 	);
		mem_free( A 	);
	}
	mem_free( A_temp	);
	mem_free( G 		);
	mem_free( errors 	);
	mem_free( perm 		);
	mem_free( nn_out	);
	mem_free( nn_t_out 	);
	mem_free( work 		);
	mem_free( iwork		);
	mem_free( S			);
	NN_API_ERROR = NN_API_SUCCESS;
	
	return 0;
error:
	destroy_nn( test_nn );

	/* clean memory and exit */
	if(r_samples != NULL)
	{
		mem_free(r_samples[0]);
		mem_free(r_samples);
	}

	if(r_targets != NULL)
	{
		mem_free(r_targets[0]);
		mem_free(r_targets);
	}
	
	if(J != NULL)
	{
		mem_free( J[0] 	);
		mem_free( J 	);
	}
	if(A != NULL)
	{
		mem_free( A[0] 	);
		mem_free( A 	);
	}
	mem_free( G 		);
	mem_free( errors 	);
	mem_free( perm 		);
	mem_free( nn_out	);
	mem_free( nn_t_out 	);
	mem_free( work 		);
	mem_free( iwork		);
	mem_free( S			);
	
	NN_API_ERROR = NN_API_TRAIN;
	return -1;
}


/* ===================================== 
 * = Neural network training functions =
 * =====================================
 */
int32_t nn_train2(net_t * nn, double **samples, double ** targets, int32_t num_samples, int32_t max_epochs, double min_error, double train_perc )
{
	double ** J = NULL, ** A = NULL, * A_temp = NULL, * G = NULL, * errors = NULL, * nn_out = NULL, * nn_t_out = NULL, * work = NULL, * S = NULL;
	double niu = 0.001, current_error = 1e20, new_error = 0, error_perc = 1- train_perc, grb = 0;
	double **r_samples = NULL, **r_targets = NULL;
	int32_t * iwork = NULL; 
	int32_t l = 0, J_ydim = 0, nlvl = 0, epoch = 0, *perm = NULL, lwork = 0, s = 0, num_outputs = 0, num_inputs = 0;
	int32_t i = 0, train_s_index = 0, error_e_index = 0;
	net_layer_t *layer = NULL;
	net_t * test_nn = NULL;
	
	/* conditions check */
	C_CHECK_CONDITION( nn == NULL, 							 NN_API_NOT_INIT  );
	C_CHECK_CONDITION( nn->init_state == false, 			 NN_API_NOT_INIT  );
	C_CHECK_CONDITION( num_samples <= 0, 					 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( max_epochs < 0 && max_epochs > 10000, NN_API_BAD_INPUT );
	C_CHECK_CONDITION( min_error < 0 && min_error > 1000,	 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( samples == NULL, 					 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( targets == NULL, 					 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( train_perc <= 0 || train_perc >= 1, 	 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( train_perc*num_samples < 10,		 	 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( error_perc*num_samples < 10,		 	 NN_API_BAD_INPUT );
	
	/* init weights */
	C_SAFE_CALL( init_nn_weights(nn) );
	
	/* compute jacobian number of lines */
	for(l = 1; l < nn->num_layers; l++){
		layer = nn->layer[l];
		J_ydim += (layer->num_inputs + 1) * layer-> num_outputs;
	}
	
	/* first and last layer : required to know the number of inputs and outputs and so allocate memory */
	layer = nn->layer[nn->num_layers-1];
	num_outputs = layer->num_outputs;
	
	layer = nn->layer[0];
	num_inputs = layer->num_outputs;
	
	/* initialize required memory */
	C_SAFE_CALL( J 		  = (double **)mem_alloc(num_outputs * num_samples	* sizeof(double  *), true) );
	C_SAFE_CALL( A 		  = (double **)mem_alloc(J_ydim     				* sizeof(double  *), true) );
	C_SAFE_CALL( G 	 	  = (double  *)mem_alloc(J_ydim		  				* sizeof(double   ), true) );
	C_SAFE_CALL( errors   = (double  *)mem_alloc(num_outputs * num_samples 	* sizeof(double   ), true) );
	C_SAFE_CALL( perm     = (int32_t *)mem_alloc(num_samples 				* sizeof(int32_t  ), true) );
	C_SAFE_CALL( nn_out   = (double  *)mem_alloc(num_outputs		   		* sizeof(double   ), true) );
	C_SAFE_CALL( nn_t_out = (double  *)mem_alloc(num_outputs		   		* sizeof(double   ), true) );
	
	C_SAFE_CALL( J[0]     	  = (double  *)mem_alloc(num_outputs * num_samples *  J_ydim * sizeof(double ), true) );
	C_SAFE_CALL( A[0]      	  = (double  *)mem_alloc(J_ydim * J_ydim  				 	 * sizeof(double ), true) );
	C_SAFE_CALL( A_temp   	  = (double  *)mem_alloc(J_ydim * J_ydim  					 * sizeof(double ), true) );
	C_SAFE_CALL( r_samples    = (double **)mem_alloc(num_samples 						 * sizeof(double*), true) );
	C_SAFE_CALL( r_samples[0] = (double  *)mem_alloc(num_samples * num_inputs 			 * sizeof(double*), true) );
	C_SAFE_CALL( r_targets    = (double **)mem_alloc(num_samples 						 * sizeof(double*), true) );
	C_SAFE_CALL( r_targets[0] = (double  *)mem_alloc(num_samples * num_outputs 			 * sizeof(double*), true) );
	
	/* calculate pointers address */
	for(l = 1; l < J_ydim; l++)
		A[l] = A[0] + l * J_ydim; 
		
	for(l = 1; l < num_outputs * num_samples; l++)
		J[l] = J[0] + l * J_ydim;
	
	for(l=1; l < num_samples; l++){
		r_samples[l] = r_samples[0] + l * num_inputs;
		r_targets[l] = r_targets[0] + l * num_outputs;
	}
	
	/* query lapack for the best size of the work area */
	nlvl = (int32_t)(floor(log2((double)J_ydim/(25.0f+1.0f)))+1.0f);

	C_SAFE_CALL( S 		= (double  *) mem_alloc( J_ydim 							 * sizeof(double ), true) );
	C_SAFE_CALL( iwork 	= (int32_t *) mem_alloc( ( 3 * J_ydim * nlvl + 11 * J_ydim ) * sizeof(int32_t), true) );  
	lwork = lpck_solve_ls_svd_pivot(J_ydim, -1, NULL, 0, S, iwork, A[0], G);
	C_SAFE_CALL( work = (double *)mem_alloc(lwork * sizeof(double), true) );
	
	/* grab scalling parameters */
	nn_grab_rescale_params( nn, samples, targets, num_samples );
	
	/* rescale samples and targets to -1, 1 range */
	for(l = 0; l < num_samples; l++){
		memcpy(r_samples[l], samples[l], num_inputs * sizeof(double));
		C_SAFE_CALL( transform_in_out_data(nn, r_samples[l], num_inputs, false) );

		memcpy(r_targets[l], targets[l], num_outputs * sizeof(double));
		for(i = 0; i < num_outputs; i++){
			r_targets[l][i] = map_min_max(r_targets[l][i], nn->ymin[i], nn->ymax[i], 0, 0, false);
		}
	}
	
	/* generate a duplicate network */
	C_SAFE_CALL( test_nn = duplicate_network( nn ) );
	
	/* copy network to a safe place */
	C_SAFE_CALL( copy_nn_data( test_nn, nn ) );
		
	/* generate a random permutation */
	random_perm(num_samples, perm);
		
	/* training start index and error end index */
	train_s_index = num_samples * error_perc + 1;
	error_e_index = num_samples * error_perc;
	
	/* disable rescaling */
	nn->use_mapping = false;
	
	/* ASSUMPTION: FIRST SAMPLES ON PERM ARE FOR ERROR CALCULATION*/
	C_SAFE_CALL( current_error = compute_JGH(test_nn, J, G, A, r_samples, r_targets, errors, nn_out, nn_t_out, perm, error_e_index, num_outputs, J_ydim) );
	
	/* compute new hessian, jacobian and gradient */
	C_SAFE_CALL(	grb = compute_JGH(test_nn, J, G, A, r_samples, r_targets, 
							&errors[train_s_index*num_outputs], nn_out, nn_t_out,
							&perm[train_s_index], 
							num_samples - train_s_index, 
							num_outputs, J_ydim);
				);
					
	/* lets train the neural network */
	while(sqrt(current_error) > min_error && epoch < max_epochs && niu < 1e20)
	{
		
		/* copy network to a safe place */
		C_SAFE_CALL( copy_nn_data( test_nn, nn ) );
		
		memcpy(A_temp, A[0], J_ydim * J_ydim * sizeof(double));
		
		/* apply niu */
		for(i = 0; i < J_ydim; i++)
			 A_temp[i*J_ydim+i] += niu ;
		
		/* call lapack to solve our system of equations */
		C_SAFE_CALL( lpck_solve_ls_svd_pivot(J_ydim, 0, work, lwork, S, iwork, A_temp, G) );
		
		/* apply changes to neural network weights (lapack gives solution in G) */
		nn_change_weights_bias(test_nn, G);
		
		/* compute new error */
		new_error = 0.0f;
		for(s = 0; s < error_e_index; s++)
		{
			/* short circuit the pointers */
			test_nn->in_layer = r_samples[perm[s]]; test_nn->out_layer = nn_out;
			
			/* compute neural network response for this input */
			C_SAFE_CALL( get_nn_response( test_nn ) );
			
			/* compute sample errors */
			for(i = 0; i < num_outputs; i++)
			{
				new_error += (r_targets[perm[s]][i] - nn_out[i]) * (r_targets[perm[s]][i] - nn_out[i]);
			}
		}
		
		
		/* check wether or not we should accept this network */
		if(new_error < current_error )
		{
			/* copy network */
			C_SAFE_CALL( copy_nn_data( nn, test_nn ) );
			current_error = new_error;
			/* compute new hessian, jacobian and gradient */
			C_SAFE_CALL(
				grb = compute_JGH(test_nn, J, G, A, r_samples, r_targets, 
								&errors[train_s_index*num_outputs], nn_out, nn_t_out,
								&perm[train_s_index], 
								num_samples - train_s_index, 
								num_outputs, J_ydim);
			);
			niu *= 0.1;
		}else{

			niu *= 10;
		}
		
		printf("epoch: %d error : %f niu : %f\n",epoch,sqrt(current_error),niu);
		
		epoch++;
	}
	
	/* clean memory and exit */
	destroy_nn( test_nn );
	
	/* enable rescaling */
	nn->use_mapping = true;
	
	if(r_samples != NULL)
	{
		mem_free(r_samples[0]);
		mem_free(r_samples);
	}

	if(r_targets != NULL)
	{
		mem_free(r_targets[0]);
		mem_free(r_targets);
	}
	
	if(J != NULL)
	{
		mem_free( J[0] 	);
		mem_free( J 	);
	}
	if(A != NULL)
	{
		mem_free( A[0] 	);
		mem_free( A 	);
	}
	mem_free( A_temp 	);
	mem_free( G 		);
	mem_free( errors 	);
	mem_free( perm 		);
	mem_free( nn_out	);
	mem_free( nn_t_out 	);
	mem_free( work 		);
	mem_free( iwork		);
	mem_free( S			);
	NN_API_ERROR = NN_API_SUCCESS;
	
	return 0;
error:
	destroy_nn( test_nn );

	/* clean memory and exit */
	if(r_samples != NULL)
	{
		mem_free(r_samples[0]);
		mem_free(r_samples);
	}

	if(r_targets != NULL)
	{
		mem_free(r_targets[0]);
		mem_free(r_targets);
	}
	
	if(J != NULL)
	{
		mem_free( J[0] 	);
		mem_free( J 	);
	}
	if(A != NULL)
	{
		mem_free( A[0] 	);
		mem_free( A 	);
	}
	mem_free( A_temp 	);
	mem_free( G 		);
	mem_free( errors 	);
	mem_free( perm 		);
	mem_free( nn_out	);
	mem_free( nn_t_out 	);
	mem_free( work 		);
	mem_free( iwork		);
	mem_free( S			);
	
	NN_API_ERROR = NN_API_TRAIN2;
	return -1;
}


int32_t nn_train3(net_t * nn, double **samples, double ** targets, int32_t num_samples, int32_t max_epochs, double min_error )
{
	double ** J = NULL, ** A = NULL, *A_temp = NULL, * G = NULL, * errors = NULL, * nn_out = NULL, * nn_t_out = NULL, * work = NULL, * S = NULL, * inv_ipiv = NULL, * inv_work = NULL;
	double niu = 0.01, current_error = 1e20, new_error = 0,  alpha = 0, beta = 1, gamma = 0, Ew = 0, Ed = 0, trace = 0;
	double **r_samples = NULL, **r_targets = NULL;
	int32_t * iwork = NULL; 
	int32_t l = 0, J_ydim = 0, nlvl = 0, epoch = 0, *perm = NULL, lwork = 0, s = 0, num_outputs = 0, num_inputs  =0;
	int32_t i = 0;
	net_layer_t *layer = NULL;
	net_t * test_nn = NULL;
	
	/* conditions check */
	C_CHECK_CONDITION( nn == NULL, 							 NN_API_NOT_INIT  );
	C_CHECK_CONDITION( nn->init_state == false, 			 NN_API_NOT_INIT  );
	C_CHECK_CONDITION( num_samples <= 0, 					 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( max_epochs < 0 && max_epochs > 10000, NN_API_BAD_INPUT );
	C_CHECK_CONDITION( min_error < 0 && min_error > 1000,	 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( samples == NULL, 					 NN_API_BAD_INPUT );
	C_CHECK_CONDITION( targets == NULL, 					 NN_API_BAD_INPUT );
	
	/* init weights */
	C_SAFE_CALL( init_nn_weights(nn) );
	
	/* compute jacobian number of lines */
	for(l = 1; l < nn->num_layers; l++){
		layer = nn->layer[l];
		J_ydim += (layer->num_inputs + 1) * layer-> num_outputs;
	}
	
	/* first and last layer : required to know the number of inputs and outputs and so allocate memory */
	layer = nn->layer[nn->num_layers-1];
	num_outputs = layer->num_outputs;
	
	layer = nn->layer[0];
	num_inputs = layer->num_outputs;
	
	
	/* initialize required memory */
	C_SAFE_CALL( J 		  = (double **)mem_alloc(num_outputs * num_samples	* sizeof(double  *), true) );
	C_SAFE_CALL( A 		  = (double **)mem_alloc(J_ydim     				* sizeof(double  *), true) );
	C_SAFE_CALL( G 	 	  = (double  *)mem_alloc(J_ydim		  				* sizeof(double   ), true) );
	C_SAFE_CALL( errors   = (double  *)mem_alloc(num_outputs * num_samples 	* sizeof(double   ), true) );
	C_SAFE_CALL( perm     = (int32_t *)mem_alloc(num_samples 				* sizeof(int32_t  ), true) );
	C_SAFE_CALL( nn_out   = (double  *)mem_alloc(num_outputs		   		* sizeof(double   ), true) );
	C_SAFE_CALL( nn_t_out = (double  *)mem_alloc(num_outputs		   		* sizeof(double   ), true) );
	
	C_SAFE_CALL( J[0]     	  = (double  *)mem_alloc(num_outputs * num_samples *  J_ydim * sizeof(double ), true) );
	C_SAFE_CALL( A[0]      	  = (double  *)mem_alloc(J_ydim * J_ydim  				 	 * sizeof(double ), true) );
	C_SAFE_CALL( A_temp   	  = (double  *)mem_alloc(J_ydim * J_ydim  					 * sizeof(double ), true) );
	C_SAFE_CALL( r_samples    = (double **)mem_alloc(num_samples 						 * sizeof(double*), true) );
	C_SAFE_CALL( r_samples[0] = (double  *)mem_alloc(num_samples * num_inputs 			 * sizeof(double*), true) );
	C_SAFE_CALL( r_targets    = (double **)mem_alloc(num_samples 						 * sizeof(double*), true) );
	C_SAFE_CALL( r_targets[0] = (double  *)mem_alloc(num_samples * num_outputs 			 * sizeof(double*), true) );
	
	/* calculate pointers address */
	for(l = 1; l < J_ydim; l++)
		A[l] = A[0] + l * J_ydim; 
		
	for(l = 1; l < num_outputs * num_samples; l++)
		J[l] = J[0] + l * J_ydim;
	
	for(l=1; l < num_samples; l++){
		r_samples[l] = r_samples[0] + l * num_inputs;
		r_targets[l] = r_targets[0] + l * num_outputs;
	}
	
	/* query lapack for the best size of the work area */
	nlvl = (int32_t)(floor(log2((double)J_ydim/(25.0f+1.0f)))+1.0f);

	C_SAFE_CALL( S 		= (double  *) mem_alloc( J_ydim 							 * sizeof(double ), true) );
	C_SAFE_CALL( iwork 	= (int32_t *) mem_alloc( ( 3 * J_ydim * nlvl + 11 * J_ydim ) * sizeof(int32_t), true) );  
	lwork = lpck_solve_ls_svd_pivot(J_ydim, -1, NULL, 0, S, iwork, A[0], G);
	C_SAFE_CALL( work = (double *)mem_alloc(lwork * sizeof(double), true) );
	
	/* allocate space for matrix inversion */
	C_SAFE_CALL( inv_ipiv = (double *) mem_alloc( (J_ydim + 1) * sizeof(double), true ) );
	C_SAFE_CALL( inv_work = (double *) mem_alloc( J_ydim * J_ydim * sizeof(double), true ) );
	
	/* grab scalling parameters */
	nn_grab_rescale_params( nn, samples, targets, num_samples );
	
	/* rescale samples and targets to -1, 1 range */
	for(l = 0; l < num_samples; l++){
		memcpy(r_samples[l], samples[l], num_inputs * sizeof(double));
		C_SAFE_CALL( transform_in_out_data(nn, r_samples[l], num_inputs, false) );

		memcpy(r_targets[l], targets[l], num_outputs * sizeof(double));
		for(i = 0; i < num_outputs; i++){
			r_targets[l][i] = map_min_max(r_targets[l][i], nn->ymin[i], nn->ymax[i], 0, 0, false);
		}
	}	
	
	/* generate a duplicate network */
	C_SAFE_CALL( test_nn = duplicate_network( nn ) );
	
	/* copy network to a safe place */
	C_SAFE_CALL( copy_nn_data( test_nn, nn ) );
		
	/* generate a random permutation */
	random_perm(num_samples, perm);
		
	C_SAFE_CALL( Ed = compute_JGH(test_nn, J, G, A, r_samples, r_targets, errors, nn_out, nn_t_out, perm, num_samples, num_outputs, J_ydim) );
	
	/* compute Ew */
	Ew = 0.0;
	for( l = 1; l < test_nn -> num_layers; l++){
		layer = test_nn -> layer[l];
		for(i = 0; i < layer -> num_inputs * layer -> num_outputs; i++)
			Ew += layer -> weights[0][i] * layer -> weights[0][i];
		for(i = 0; i < layer->num_outputs; i++)
			Ew += layer->bias[i];
	}
	current_error = alpha * Ew + beta *Ed;
	
	/* disable rescaling */
	nn->use_mapping = false;
	
	/* lets train the neural network */
	while(sqrt(current_error) > min_error && epoch < max_epochs && niu < 1e10)
	{
		
		/* copy network to a safe place */
		C_SAFE_CALL( copy_nn_data( test_nn, nn ) );
		
		memcpy(A_temp, A[0], J_ydim * J_ydim * sizeof(double) );
		for(i=0; i < J_ydim * J_ydim; i++)
			A_temp[i] *= 2.0f * beta;
		
		/* apply niu and alpha regularization parameter */
		for(i = 0; i < J_ydim; i++)
			A_temp[i*J_ydim+i] += 2.0f*niu + 2.0f*alpha;
		
		/* call lapack to solve our system of equations */
		C_SAFE_CALL( lpck_solve_ls_svd_pivot(J_ydim, 0, work, lwork, S, iwork, A_temp, G) );
		
		/* apply changes to neural network weights (lapack gives solution in G) */
		nn_change_weights_bias(test_nn, G);
		
		/* compute new error */
		Ed = 0.0f;
		for(s = 0; s < num_samples; s++)
		{
			/* short circuit the pointers */
			test_nn->in_layer = r_samples[s]; test_nn->out_layer = nn_out;
			
			/* compute neural network response for this input */
			C_SAFE_CALL( get_nn_response( test_nn ) );
			
			/* compute sample errors */
			for(i = 0; i < num_outputs; i++)
				Ed += (r_targets[s][i] - nn_out[i])*(r_targets[s][i] - nn_out[i]);
		}
		
		/* compute Ew */
		Ew = 0.0;
		for( l = 1; l < test_nn -> num_layers; l++){
			layer = test_nn -> layer[l];
			for(i = 0; i < layer -> num_inputs * layer -> num_outputs; i++)
				Ew += layer -> weights[0][i] * layer -> weights[0][i];
			for(i = 0; i < layer->num_outputs; i++)
				Ew += layer->bias[i];
		}
		
		new_error = alpha * Ew + beta * Ed;
		
		/* check wether or not we should accept this network */
		if(new_error < current_error )
		{
			/* copy network */
			C_SAFE_CALL( copy_nn_data( nn, test_nn ) );
			current_error = new_error;
			niu *= 0.1;
				
			/* update parameters */
			memcpy(A_temp, A[0], J_ydim * J_ydim * sizeof(double) );
			//for(i=0; i < J_ydim * J_ydim; i++)
			//	A_temp[i] *= 2.0f * beta;
		
			/* apply niu and alpha regularization parameter */
			//for(i = 0; i < J_ydim; i++)
			//	A_temp[i*J_ydim+i] += 2.0f*alpha;
			
			C_SAFE_CALL( lpck_get_inv(J_ydim, inv_work, J_ydim * J_ydim, inv_ipiv, A_temp) );
			
			/* compute trace of the inverse */
			trace = 0.0;
			for(i = 0; i < J_ydim; i++)
				trace += A_temp[i*J_ydim+i];
			
			gamma = (double)(J_ydim) - (alpha * trace);
			beta =fabs(((double)(num_samples) - gamma)/(2.0f*Ed));
			alpha = (double)J_ydim /(2.0f*Ew+trace);
			
			/* compute new hessian, jacobian and gradient */
			C_SAFE_CALL( current_error = compute_JGH(test_nn, J, G, A, r_samples, r_targets, 
								errors, nn_out, nn_t_out, perm, num_samples, num_outputs, J_ydim);
						);
						
		}else{
			niu *= 10;
		}
		
		printf("epoch: %d  alpha : %f beta : %f gamma : (%f ,%d, %d) trace : %f error : %f (%f, %f) niu : %f\n",epoch,alpha, beta, gamma, J_ydim,num_samples, trace,sqrt(current_error), Ed, Ew,niu);
		
		epoch++;
	}
	
	/* clean memory and exit */
	destroy_nn( test_nn );
	
	/* enable rescaling */
	nn->use_mapping = true;
	
	if(r_samples != NULL)
	{
		mem_free(r_samples[0]);
		mem_free(r_samples);
	}

	if(r_targets != NULL)
	{
		mem_free(r_targets[0]);
		mem_free(r_targets);
	}
	
	if(J != NULL)
	{
		mem_free( J[0] 	);
		mem_free( J 	);
	}
	if(A != NULL)
	{
		mem_free( A[0] 	);
		mem_free( A 	);
	}
	mem_free( A_temp    );
	mem_free( G 		);
	mem_free( errors 	);
	mem_free( perm 		);
	mem_free( nn_out	);
	mem_free( nn_t_out 	);
	mem_free( work 		);
	mem_free( iwork		);
	mem_free( S			);
	mem_free( inv_ipiv  );
	mem_free( inv_work  );
	
	NN_API_ERROR = NN_API_SUCCESS;
	
	return 0;
error:
	destroy_nn( test_nn );

	/* clean memory and exit */
	if(r_samples != NULL)
	{
		mem_free(r_samples[0]);
		mem_free(r_samples);
	}

	if(r_targets != NULL)
	{
		mem_free(r_targets[0]);
		mem_free(r_targets);
	}
	
	if(J != NULL)
	{
		mem_free( J[0] 	);
		mem_free( J 	);
	}
	if(A != NULL)
	{
		mem_free( A[0] 	);
		mem_free( A 	);
	}
	mem_free( G 		);
	mem_free( errors 	);
	mem_free( perm 		);
	mem_free( nn_out	);
	mem_free( nn_t_out 	);
	mem_free( work 		);
	mem_free( iwork		);
	mem_free( S			);
	
	NN_API_ERROR = NN_API_TRAIN3;
	return -1;
}
/* create a network with the same characteristics of the one that has been given */
net_t * duplicate_network(net_t * nn )
{
	int32_t * layers_func = NULL, * layers_conf = NULL;
	int32_t i = 0;
	net_t * new_nn = NULL;
	net_layer_t * layer;
	C_CHECK_CONDITION( nn == NULL, 					NN_API_BAD_INPUT );
	C_CHECK_CONDITION( nn -> layer == NULL, 		NN_API_BAD_INPUT );  
	C_CHECK_CONDITION( nn -> init_state == false, 	NN_API_BAD_INPUT );  
	C_CHECK_CONDITION( nn -> num_layers == 0, 		NN_API_BAD_INPUT );
	  
	/* first allocate temporary memory */   
	C_SAFE_CALL( layers_func = (int32_t *) mem_alloc( nn->num_layers * sizeof(int32_t), true ) );
	C_SAFE_CALL( layers_conf = (int32_t *) mem_alloc( nn->num_layers * sizeof(int32_t), true ) );
	 
	/* get the parameters from this network */
	for(i = 0; i < nn->num_layers; i++)
	{
		layer = nn -> layer[i];
		C_CHECK_CONDITION( layer == NULL, NN_API_MALFORMED_NN );  
		layers_conf[i] = layer->num_outputs;
		layers_func[i] = layer->eval_funct;
		
	}
	
	/* call network initializer */ 
	C_SAFE_CALL( new_nn = create_nn(nn->num_layers, layers_conf, layers_func ) );
	
	/* should we also copy the information? */
	
	/* input layer rescale parameters */
	for(i = 0; i < (nn->layer[0])->num_outputs; i++){
		new_nn -> xmin[i] = nn -> xmin[i];
		new_nn -> xmax[i] = nn -> xmax[i];
	}
	for(i = 0; i < (nn->layer[nn->num_layers-1])->num_outputs; i++){
		new_nn -> ymin[i] = nn -> ymin[i];	
		new_nn -> ymax[i] = nn -> ymax[i];
	}
	
	/* free auxiliary memory areas */
	mem_free(layers_func);
	mem_free(layers_conf);

	NN_API_ERROR = NN_API_SUCCESS;
	return new_nn;
error:

	/* free auxiliary memory areas */
	mem_free(layers_func);
	mem_free(layers_conf);

	NN_API_ERROR = NN_API_DUPLICATE_NN;
	return NULL;
}

/* copy data from one network into another */
int32_t copy_nn_data(net_t *dest, net_t *src)
{
	net_layer_t * l_src = NULL, * l_dest = NULL;
	int32_t l = 0;
	
	C_CHECK_CONDITION( dest -> num_layers != src -> num_layers, NN_API_COPY_DIFF );
	C_CHECK_CONDITION( src -> num_layers == 0, 					NN_API_COPY_DIFF );
	
	for( l = 1; l < src -> num_layers; l++)
	{
		l_src = src->layer[l];
		l_dest = dest->layer[l];
		
		C_CHECK_CONDITION( l_src->num_inputs  != l_dest->num_inputs,  NN_API_COPY_DIFF );
		C_CHECK_CONDITION( l_src->num_outputs != l_dest->num_outputs, NN_API_COPY_DIFF );
	
		/* proceed with the copy */
		memcpy(l_dest->weights[0], l_src->weights[0], l_src->num_inputs * l_src->num_outputs * sizeof(double));
		memcpy(l_dest->bias, l_src->bias, l_src->num_outputs * sizeof(double));
	
	}
	dest->use_mapping = src->use_mapping;

	
	NN_API_ERROR = NN_API_SUCCESS;
	return 0;
error:
	NN_API_ERROR = NN_API_COPY;
	return -1;
}


void nn_grab_rescale_params(net_t * nn, double ** samples, double ** targets, int32_t num_samples )
{
	int32_t i = 0, j = 0;
	net_layer_t * layer = NULL;
	double min = 0, max = 0;
	layer = nn->layer[0];
	
	/* first find maximum and minimum of samples */
	min = samples[0][0]; max = min;
	
	for(i = 0; i < layer->num_outputs; i++)
	{
		for(j = 0; j < num_samples; j++ )
		{
			if(max < samples[j][i]) 
				max = samples[j][i];
			if(min > samples[j][i])
				min = samples[j][i]; 
		}
		/* store input parameters */
		nn->xmin[i] = min;
		nn->xmax[i] = max;
	}
	
	/* do it for targets*/
	layer = nn->layer[nn->num_layers-1];
	min = targets[0][0]; max = min;
	for(i = 0; i < layer->num_outputs; i++)
	{
		for(j = 0; j < num_samples; j++ )
		{
			if(max < targets[j][i]) 
				max = targets[j][i];
			if(min > targets[j][i])
				min = targets[j][i]; 
		}
		/* store output parameters */
		nn->ymin[i] = min;
		nn->ymax[i] = max;
	}
}

/*
 * Change weights and bias based on the values provided in G 
 */
void nn_change_weights_bias(net_t * nn, double * G)
{
	net_layer_t * layer = NULL;
	int32_t l = 0, i = 0, j = 0, ct = 0;
	
	/* for each layer */
	for(l = 1; l < nn->num_layers; l++)
	{
		layer = nn->layer[l];
		
		for(i = 0; i < layer->num_outputs; i++)
		{	
			/* weights */
			for(j = 0; j < layer->num_inputs; j++)
			{
				 layer->weights[i][j] -= G[ct++];
			}
			
			/* bias */
			layer->bias[i] -= G[ct++];
		}
	}	
	
}

int32_t init_nn_weights(net_t * nn)
{
	int32_t i = 0, l = 0, n = 0;
	net_layer_t * layer = NULL;
	double norm_weights = 0.0f, norm_bias = 0.0f, beta = 0.0f;
	double f_range_up = 0.0f, f_range_down = 0.0f;
	
	C_CHECK_CONDITION( nn->num_layers == 0, NN_API_BAD_INPUT );

	for(l = 1; l < nn->num_layers; l++)
	{
		layer = nn->layer[l];
		
		/* scale random values according to the function used */
		switch(layer->eval_funct)
		{
			case      NN_LOGSIG: { f_range_up =  1.0f; f_range_down =  0.0f; break; }
			case      NN_TANSIG: { f_range_up =  1.0f; f_range_down = -1.0f; break; }
			default:  			 { f_range_up =  1.0f; f_range_down = -1.0f; break; }
		}
		
		beta = 0.7f*pow((double)layer->num_outputs, 1/(double)layer->num_inputs);
		
		/* for each neuron in the layer */
		norm_weights = 0.0f; norm_bias = 0.0f;
		for( n = 0; n < layer->num_outputs; n++)
		{
			for(i = 0; i < layer->num_inputs; i++ )
			{
				layer->weights[n][i] = (double)(mt_random()) * RANDMAXV * (f_range_up - f_range_down) - f_range_down;
				norm_weights += layer->weights[n][i] * layer->weights[n][i];
			}
			
			/* randomize bias */
			norm_weights = sqrt(norm_weights);
			norm_bias = sqrt(norm_bias);
			for(i = 0; i < layer->num_inputs; i++ )
				layer->weights[n][i] *=  (beta/norm_weights);
		
			// generate bias
			if(layer->weights[n][0]>0)
				layer->bias[n] = beta * (-1.0f + (double)n*2/(double)(layer->num_outputs));
			else
				layer->bias[n] = -1.0*beta * (-1.0f + (double)n*2/(double)(layer->num_outputs));
		}	

	}

	NN_API_ERROR = NN_API_SUCCESS;
	return 0;
	
error:

	NN_API_ERROR = NN_API_WEIGHT_INIT;
	return -1;
}

int32_t lpck_solve_nxn_ls(int32_t J_ydim, int32_t query, int32_t * work, int32_t lwork, double *eq_matrix, double * eq_sol)
{
	int32_t col_size = J_ydim;
	int32_t car = 1;
	int32_t info = 0;
	if(query == -1)
	{
		return J_ydim;
	}else{
		dgesv_( &col_size, &car, eq_matrix, &col_size, work, eq_sol, &col_size, &info );
	
		/* Check for the exact singularity */
    	C_CHECK_CONDITION(info != 0, NN_API_LS_SOLVE);
    }
    NN_API_ERROR = NN_API_SUCCESS;
    return info;
    
	
error:
	if( info > 0 ) {
                printf( "\t\tThe diagonal element of the triangular factor of A,\n" );
                printf( "\t\tU(%i,%i) is zero, so that A is singular;\n", info, info );
                printf( "\t\tthe solution could not be computed.\n" );
                exit( 1 );
        }
    NN_API_ERROR = NN_API_LS;
	return -1;
}


int32_t lpck_solve_ls_svd_pivot(int32_t J_ydim, int32_t query, double * work, int32_t lwork, double * S, int32_t * iwork,  double *eq_matrix, double * eq_sol)
{
	int32_t col_size = J_ydim;
	int32_t nrhs = 1;
	int32_t info = 0;
	int32_t rank = 0;
	double work_size = 0;
	double rcond = -1;			/* default machine precision */
	
	if(query == -1)
	{
		dgelsd_( &col_size,		/* m dimension */ 
				&col_size,		/* n dimension */
				&nrhs, 			/* number of right hand sided */
				eq_matrix, 		/* A matrix */
				&col_size, 		/* leading dimension of A */
				eq_sol,			/* B vector */
				&col_size,		/* leading dimension of B */
				S, 				/* array with M columns */
				&rcond, 		/* which precision? */
				&rank, 			/* the matrix rank */
				&work_size,		/* query workspace size */
				&query,			/* the query */
				iwork, 			/* workspace array */
				&info );
		return (int32_t)work_size;
	}else{
	
		dgelsd_( &col_size,		/* m dimension */ 
				&col_size,		/* n dimension */
				&nrhs, 			/* number of right hand sided */
				eq_matrix, 		/* A matrix */
				&col_size, 		/* leading dimension of A */
				eq_sol,			/* B vector */
				&col_size,		/* leading dimension of B */
				S, 				/* array with M columns */
				&rcond, 		/* which precision? */
				&rank, 			/* the matrix rank */
				work,			/* query workspace size */
				&lwork,			/* the query */
				iwork, 			/* workspace array */
				&info );
		/* Check for the exact singularity */
    	C_CHECK_CONDITION(info != 0, NN_API_LS_SOLVE);
    }
    NN_API_ERROR = NN_API_SUCCESS;
    return info;
    
	
error:
	if( info > 0 ) {
        printf( "\t\tThe diagonal element of the triangular factor of A,\n" );
        printf( "\t\tU(%i,%i) is zero, so that A is singular;\n", info, info );
        printf( "\t\tthe solution could not be computed.\n" );
    }
       
    
    NN_API_ERROR = NN_API_LS;
	return -1;
}


int32_t lpck_solve_ls(int32_t J_ydim, int32_t query, double * work, int32_t lwork, double * eq_matrix,  double * eq_sol)
{	
	char  N = 'N';
	int32_t col_size = J_ydim;
	int32_t car = 1;
	int32_t info = 0;
	double work_size = 7000;
	
	
	if (query==-1)
	{
		
		dgels_(  &N,			         /* use matrix M or its transpose */
			    &col_size,           /* number of lines of matrix M   */
				&col_size,           /* number of rows of matrix M    */
				&car,                /* number of colums of B and x   */
				eq_matrix,           /* matrix M                      */
				&col_size,           /* size -> max(1,vector_size)    */
				eq_sol,              /* sol vector / out vector       */
				&col_size,           /* LDB                           */
				&work_size,          /* work array                    */
				&query,              /* LWORK                         */
				&info);
		return (int32_t)work_size;
	}else{
		
		dgels_(  &N,				     /* use matrix M or its transpose */
			    &col_size,           /* number of lines of matrix M   */
				&col_size,           /* number of rows of matrix M    */
				&car,                /* number of colums of B and x   */
				eq_matrix,           /* matrix M                      */
				&col_size,           /* size -> max(1,vector_size)    */
				eq_sol,        		 /* sol vector / out vector       */
				&col_size,           /* LDB                           */
				work,				 /* work array                    */
				&lwork,              /* LWORK                         */
				&info);
		
		
		C_CHECK_CONDITION(info != 0, NN_API_LS_SOLVE);
	}

	NN_API_ERROR = NN_API_SUCCESS;
	return info;
error:
	if( info > 0 )
	{
        printf( "\t\tThe diagonal element %i of the triangular factor ", info );
        printf( "\t\tof A is zero, so that A does not have full rank;\n" );
        printf( "\t\tthe least squares solution could not be computed.\n" );
    }else{
    	printf("Lwork value %d\n", lwork);
    	
    	
    }
    NN_API_ERROR = NN_API_LS;
	return -1;
}


int32_t lpck_get_inv(int32_t J_ydim, double * work, int32_t lwork, double * ipiv, double * eq_matrix)
{	
	int32_t col_size = J_ydim;
	int32_t info = 0;
	int32_t i = 0;
	
	dgetrf_(&col_size,&col_size, eq_matrix, &col_size, ipiv, &info);

	if(info!= 0){
		goto error;	
	}
	dgetri_(&col_size, eq_matrix, &col_size, ipiv, work, &lwork, &info);
	
	if(info!= 0){
		goto error;	
	}
	
	if(info!= 0)
		goto error;
		
	NN_API_ERROR = NN_API_SUCCESS;
	return info;
error:
	if(info > 0)
		printf("decomposition successful but U is singular, u_%d%d=0\n", i,i);
	else
		printf("decomposition not successful %dth parameter has an illegal value\n", i);			
	
	NN_API_ERROR = NN_API_GET_INV;
	return -1;
}


void random_perm(int32_t num_p, int32_t * ranpat)
{
	unsigned int p, np, op;
	unsigned int r;
	for( p = 0 ; p < (unsigned int)num_p; p++ ) { ranpat[p] = (unsigned int)p; } 

	for( p = 0 ; p < (unsigned int)num_p; p++) 
	{	
		r =  (unsigned int)( mt_random() * 1.0 * ( RANDMAXV  * (num_p - p )));
		np = p + r ;
		op = ranpat[p]; 
		ranpat[p] = ranpat[np]; 
		ranpat[np] = op;
	}
}

void init_mt(void)
{
    int32_t i = 0;
    srand((int32_t)time(NULL));
    /* init mersenne twister */
	mt_state[0]= rand() & 0xffffffffUL;
    for (i = 1; i < MT_N; i++)
	{
            mt_state[i] = (1812433253UL * (mt_state[i - 1] ^ (mt_state[i-1] >> 30)) + i); 
        	/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
       		/* In the previous versions, MSBs of the seed affect   */
        	/* only MSBs of the array mt[].                        */
        	/* 2002/01/09 modified by Makoto Matsumoto             */
        	mt_state[i] &= 0xffffffffUL;
        	/* for >32 bit machines */
    }
	
	mt_constants[0] = 0x0L;
	mt_constants[1] = 0x9908b0dfUL;
	mt_state_idx = MT_N+1;
}


unsigned long mt_random(void){
	unsigned long y;
	if(rand_state == false){ init_mt(); rand_state = true; } 
	/* Updating state */
	if( mt_state_idx >= MT_N){
		int k;
		for(k = 0; k < MT_N - MT_M; k++)
		{
			y = ( mt_state[ k ] & MT_UPPER_MASK ) | ( mt_state[ k + 1 ] & MT_LOWER_MASK );
			mt_state[ k ] = mt_state[ k + MT_M ] ^ ( y>> 1) ^ mt_constants[ y & 0x1UL ];
		}
		for(; k < MT_N - 1; k++){
			y = ( mt_state[ k ] & MT_UPPER_MASK ) | ( mt_state[ k + 1 ] & MT_LOWER_MASK );
			mt_state[ k ] = mt_state[ k + ( MT_M - MT_N ) ] ^ ( y>> 1) ^ mt_constants[ y & 0x1UL ];
		}
		y = (mt_state[ MT_N - 1 ] & MT_UPPER_MASK ) | ( mt_state[0] & MT_LOWER_MASK );
		mt_state[ MT_N - 1 ] = mt_state[ MT_M - 1 ] ^ (y >> 1) ^ mt_constants[ y & 0x1UL ];
		mt_state_idx = 0;
	}
	/* Generating */
	y = mt_state[ mt_state_idx++ ];
	/* tempering */
	y ^= (y >> 11);
	y ^= (y << 7 )  & 0x9d2c5680UL; 
	y ^= (y << 15 ) & 0xefc60000UL;
	y ^= (y >> 18);
	return y;  
} 

