/* model_cyclic.h													*/

/* M. Walker July 2011 (not much left of PMD's original code)		*/

#ifndef _MODEL_CYCLIC_H
#define _MODEL_CYCLIC_H

#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "cyclic_utils.h"

/* Added by MAW 15/07/2011. Streamlined version of PMD model		*/
int make_model_cs(CS *cs_model, const struct filter_freq *hf,
				  const struct profile_harm *s0,
				  const struct cyclic_work *w);


/* Construct the least-squares fit of the pulse-profile (harmonics)	*/
/* to the data, given a model filter. Output s0 must be allocated	*/
/* before calling this function.  Only 1-pol for now.				*/
int optimise_profile(struct profile_harm *s0, const CS *cs,
					 const struct filter_freq *hf,
					 const struct cyclic_work *w);
	

/* Function to compute the gradient of the sum-of-squares			*/
/* with respect to the variables being optimised					*/
/* This function added by MAW 16/7/2011 for use with NLOPT			*/
int merit_gradient(struct filter_freq *gradient, const CS *cs, 
				   const struct filter_freq *hf,
				   const struct profile_harm *s0, 
				   const struct cyclic_work *w);


/* Construct an estimate of the filter H(freq) by iteration			*/
/* Added by MAW 19/7/2011. Only 1-pol for now.						*/
int construct_filter_freq(struct filter_freq *hf,
						  const struct profile_harm *s0,
						  const CS *cs,
						  const struct cyclic_work *w);
	

#endif
