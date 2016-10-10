#include <stdio.h>
#include <time.h>
#include "neural_network.h"
#include <sys/time.h>
#include <math.h>

#define NUM_POINTS 100
#define NUM_T_POINTS 120
#define pi 3.141592653589793
int main(int argc, char * argv[])
{
	#define TIMER_START() gettimeofday(&tv1, NULL)
	#define TIMER_STOP() \
    	gettimeofday(&tv2, NULL);    \
    	timersub(&tv2, &tv1, &tv);   \
    	time_delta = (float)tv.tv_sec + tv.tv_usec / 1000000.0
 	
    struct timeval tv1, tv2, tv;
    float time_delta;

	/* lets test the neural network in two situations */
	double inputs[NUM_POINTS];
	double targets[NUM_POINTS];
	double *ptr_in[NUM_POINTS];
	double *ptr_out[NUM_POINTS];
	
	int32_t i = 0;
	double  result = 0;
	/* neural network description */
	int32_t layers_conf[] = {		 1,  	   60,  	   1};
	int32_t layers_func[] = {  NN_NONE, NN_TANSIG, NN_LINEAR};
	int32_t num_layers = 3;
	
	TIMER_START(); 
	
	/* init neural network */
	net_t * nn = create_nn(num_layers, layers_conf, layers_func );
	
	if(nn == NULL){ goto error; }

	/* fill data*/
	for(i = 0; i < NUM_POINTS; i++){
		inputs[i] = 2.0f*pi*(double)i/NUM_POINTS; 	
		targets[i] =10.0f*sin(inputs[i])+1.0f*(2.0f*mt_random()*RANDMAXV -1.0f);

		ptr_in[i] = inputs + i;
		ptr_out[i] = targets + i;
	}

	
	
	/* train neural network */
	nn_train(nn,  ptr_in, ptr_out, NUM_POINTS, 1000, 0.00001);
	
	/* test neural network */
	nn->out_layer = &result;
	
	printf("in : target nn\n");
	for(i = 0; i < NUM_POINTS; i++)
	{
		nn->in_layer = ptr_in[i];
		get_nn_response( nn );
		
		printf("%f : %f %f\n", ptr_in[i][0], ptr_out[i][0], result);
	}
	
	/* destroy neural network */
	destroy_nn( nn );
	TIMER_STOP();
		
	printf("elapsed time: %f s\n", time_delta );	
error:
	return 0;
}
