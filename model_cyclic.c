/* model_cyclic.c													*/

/* M. Walker July 2011 (not much left of PMD's original code)		*/

#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "cyclic_utils.h"
#include "model_cyclic.h"

int make_model_cs(CS *cs_model, const struct filter_freq *hf,
				  const struct profile_harm *s0,
				  const struct cyclic_work *w) {

/* Construct a model cyclic spectrum given H(freq) and reference	*/
/* pulse-profile harmonics. Only 1-pol for now.						*/
/* This function returns the model CS for alpha >= 0				*/
/* For -ve alpha, take the complex conjugate of these values		*/
	
    /* Check dimensions												*/
    if (cs_model->npol!=1 || cs_model->nchan!=hf->nchan || 
		cs_model->nharm!=s0->nharm) { 
		printf("make_model_cs : Incompatible dimensions\n");
		return(-1); }
	
	/* Create the intrinsic cyclic spectrum							*/
	/* Currently this has no dependence on radio frequency			*/
	profile2cs(cs_model, s0);
	/* Propagate this spectrum through the filter in two steps		*/
	
	/* First multiply by the +ve sheared filter	array				*/
	/* Create a "cyclic spectrum" array filled with filter hf		*/
	struct cyclic_spectrum cs_tmp;
	cs_copy_parms(cs_model, &cs_tmp);
	cyclic_alloc_cs(&cs_tmp);
	filter2cs(&cs_tmp, hf);
	/* Shear the filter spectrum by +alpha/2						*/
	double shear = 0.5;
	cyclic_shear_cs(&cs_tmp, shear, w);
	/* Multiply the intrinsic spectrum by this sheared array		*/
	cs_multiply(&cs_tmp, cs_model);

	/* Now multiply by the conjugate of -ve shifted filter			*/
	/* Recreate a "cyclic spectrum" array filled with filter hf		*/
	filter2cs(&cs_tmp, hf);
	/* Shear the filter spectrum by -alpha/2						*/
	shear = -0.5;
	cyclic_shear_cs(&cs_tmp, shear, w);
	/* Form the complex conjugate									*/
	cs_conjugate(&cs_tmp);
	/* Multiply the intrinsic spectrum by this sheared array		*/
	/* The result is the model cyclic spectrum						*/
	cs_multiply(&cs_tmp, cs_model);
		
	/* Free-up temporary cyclic spectrum and return					*/
	cyclic_free_cs(&cs_tmp);		
    return(0);
}

int optimise_profile(struct profile_harm *s0, const CS *cs,
					 const struct filter_freq *hf,
					 const struct cyclic_work *w) {

/* Construct the optimum (least-squares) estimate of the pulse		*/
/* profile (harmonics), given the data and a model filter.			*/
/* Only 1-pol for now.												*/
		
    /* Check dimensions												*/
    if (cs->npol!=1 || cs->nchan!=hf->nchan || cs->nharm!=s0->nharm) { 
		printf("optimise_profile : Incompatible dimensions\n");
		return(-1); }
			
	/* Allocate two temporary cyclic spectral arrays				*/
	struct cyclic_spectrum cs_tmp1;
	struct cyclic_spectrum cs_tmp2;
	cs_copy_parms(cs, &cs_tmp1);
	cs_copy_parms(cs, &cs_tmp2);
	cyclic_alloc_cs(&cs_tmp1);
	cyclic_alloc_cs(&cs_tmp2);
		
	/* Determine the (conjugate of) the effect of the filter on		*/
	/* the CS. Two steps. First form the +ve shifted filter array.	*/
	/* Create a "cyclic spectrum" array filled with filter hf		*/
	filter2cs(&cs_tmp1, hf);
	/* Shear the filter spectrum by +alpha/2						*/
	double shear = 0.5;
	cyclic_shear_cs(&cs_tmp1, shear, w);
	/* Form the complex conjugate									*/
	cs_conjugate(&cs_tmp1);
	
	/* Now form the -ve shifted filter array						*/
	/* Recreate a "cyclic spectrum" array filled with filter hf		*/
	filter2cs(&cs_tmp2, hf);
	/* Shear the filter spectrum by -alpha/2						*/
	shear = -0.5;
	cyclic_shear_cs(&cs_tmp2, shear, w);
	/* And multiply by the +ve shifted array						*/
	cs_multiply(&cs_tmp1, &cs_tmp2);

	/* Copy the result into cs_tmp1									*/
	cs_copy_data(&cs_tmp2, &cs_tmp1);
	/* Form the complex conjugate									*/
	cs_conjugate(&cs_tmp2);
	
	/* Now form the arrays cs H(-)H(+)*  and |H(-)|^2 |H(+)|^2		*/
	cs_multiply(&cs_tmp1, &cs_tmp2);
	cs_multiply(cs, &cs_tmp1);

	/* Scrunch both CS arrays in the frequency direction			*/
	struct profile_harm s_tmp;
	s_tmp.nharm = s0->nharm;
	profile_alloc_harm(&s_tmp);
	cyclic_fscrunch_cs(s0, &cs_tmp1);
	cyclic_fscrunch_cs(&s_tmp, &cs_tmp2);

	/* Ratio these two arrays to get the optimised profile			*/
	s0->data[0] = 0.0 + I * 0.0;
	float denominator = 0.0;
	int ih;
	for (ih=1; ih<s0->nharm; ih++) {
		denominator = creal(s_tmp.data[ih]);
		if (denominator>0.0) {
			s0->data[ih] /= denominator;
		}
		else {
			s0->data[ih] = 0.0 + I * 0.0;
		}
	}
	
	/* Free-up temporary arrays and return							*/
	profile_free_harm(&s_tmp);
	cyclic_free_cs(&cs_tmp1);
	cyclic_free_cs(&cs_tmp2);	
    return(0);
}

int merit_gradient(struct filter_freq *gradient, const CS *cs, 
				   const struct filter_freq *hf, 
				   const struct profile_harm *s0, 
				   const struct cyclic_work *w) {
	
/* This function added by MAW 16/7/2011 for use with NLOPT			*/
/* Evaluates the gradient of the merit function with respect to		*/
/* the coefficients of the filter function H(freq)					*/
	
    /* Only valid for 1-pol data at present							*/
    if (cs->npol!=1 || cs->nchan!=gradient->nchan ||
		cs->nchan!=hf->nchan || cs->nharm!=s0->nharm) { 
		printf("merit_gradient : Incompatible dimensions\n");
		exit(1); }
	
	/* Allocate temporary CS structs								*/
	struct cyclic_spectrum cs_res;
	struct cyclic_spectrum cs_tmp1;
	struct cyclic_spectrum cs_tmp2;
	cs_copy_parms(cs, &cs_res);
	cs_copy_parms(cs, &cs_tmp1);
	cs_copy_parms(cs, &cs_tmp2);
	cyclic_alloc_cs(&cs_res);
	cyclic_alloc_cs(&cs_tmp1);
	cyclic_alloc_cs(&cs_tmp2);
	
	/* Construct the current model cyclic spectrum from hf and s0	*/
	make_model_cs(&cs_res, hf, s0, w);
	/* And form the residual ( = model - data )						*/
	cs_subtract(cs, &cs_res);
	
	/* Need to handle +ve alpha and -ve alpha separately, so copy	*/
	/* the residual cyclic spectrum to save re-computing			*/
	/* Positive values of alpha first								*/
	cs_copy_data(&cs_res, &cs_tmp1);
	/* Shear the residual											*/
	double shear = -0.5;
	cyclic_shear_cs(&cs_tmp1, shear, w);
	/* Make a filter array											*/
	filter2cs(&cs_tmp2, hf);
	/* And shear the filter array									*/
	shear = -1.0;
	cyclic_shear_cs(&cs_tmp2, shear, w);
	/* Form the product of sheared residuals with sheared filters	*/
	cs_multiply(&cs_tmp1, &cs_tmp2);
	/* For each channel, sum over all alpha where estimate is valid	*/
	/* Only one polarisation at present								*/
	int ih, ic, ihlim, ip=0;
	for (ic=0; ic<cs->nchan; ic++) {
		gradient->data[ic] = 0.0 + I * 0.0;
		ihlim = harm_limit_cs_shear_minus(ic,cs);
		for (ih=1; ih<ihlim; ih++) {
			fftwf_complex *csval = get_cs(&cs_tmp2,ih,ip,ic);
			gradient->data[ic] += 4.0 * (*csval) * conj(s0->data[ih]);
		}
	}
	
	/* Now repeat the above for negative values of alpha			*/		
	cs_copy_data(&cs_res, &cs_tmp1);
	/* Shear the residual											*/
	shear = 0.5;
	cyclic_shear_cs(&cs_tmp1, shear, w);
	/* The CS for -ve alpha is conjugate of that for +ve alpha		*/
	cs_conjugate(&cs_tmp1);
	/* Make a filter array											*/
	filter2cs(&cs_tmp2, hf);
	/* And shear the filter array									*/
	shear = 1.0;
	cyclic_shear_cs(&cs_tmp2, shear, w);
	/* Form the product of sheared residuals with sheared filters	*/
	cs_multiply(&cs_tmp1, &cs_tmp2);
	/* For each channel, sum over all alpha where estimate is valid	*/
	/* Only one polarisation at present								*/
	for (ic=0; ic<cs->nchan; ic++) {
		ihlim = harm_limit_cs_shear_plus(ic, cs);
		for (ih=1; ih<ihlim; ih++) {
			fftwf_complex *csval = get_cs(&cs_tmp2,ih,ip,ic);
			gradient->data[ic] += 4.0 * (*csval) * (s0->data[ih]);
		}
	}
	
	/* Free-up the arrays allocated herein							*/
	cyclic_free_cs(&cs_res);
	cyclic_free_cs(&cs_tmp1);
	cyclic_free_cs(&cs_tmp2);	
    return(0);
}


int construct_filter_freq(struct filter_freq *hf,
						  const struct profile_harm *s0,
						  const CS *cs, const struct cyclic_work *w) {
	
	/* Construct an estimate of the filter H(freq) by iteration		*/
	/* Only 1-pol for now.											*/
	
    /* Check dimensions												*/
    if (cs->npol!=1 || cs->nchan!=hf->nchan || cs->nharm!=s0->nharm) { 
		printf("optimise_profile : Incompatible dimensions\n");
		return(-1); }
	
	extern int verbose;
	const int stepsize=4; /* No. of chans to step per iteration		*/
	
	/* Useful abbreviations											*/
	const int nc = cs->nchan;
	const int nh = cs->nharm;
		
	/* Find rc : |cs(ichan=rc,iharm=1)| is largest					*/
	int rc = maximum_cs1(cs);
	
	/* Want shifted CS for both +ve and -ve alpha					*/
	struct cyclic_spectrum cs_pos; /* For alpha > 0					*/
	struct cyclic_spectrum cs_neg; /* For alpha < 0					*/
	cs_copy_parms(cs, &cs_pos);
	cs_copy_parms(cs, &cs_neg);
	cyclic_alloc_cs(&cs_pos);
	cyclic_alloc_cs(&cs_neg);
	
	/* Multiply the input cyclic spectrum by conj(s0)				*/
	profile2cs(&cs_pos, s0);
	cs_conjugate(&cs_pos);
	cs_multiply(cs, &cs_pos);
	cs_copy_data(&cs_pos, &cs_neg);
	cs_conjugate(&cs_neg);
	/* Shear both +ve and -ve alpha CS arrays by -alpha/2			*/
	double shear = -0.5;
	cyclic_shear_cs(&cs_pos, shear, w);
	shear = 0.5;
	cyclic_shear_cs(&cs_neg, shear, w);
	
	/* Make a new profile(harmonic) array equal to s0 * conj(s0)	*/
	struct profile_harm	s2;
	s2.nharm = nh;
	profile_alloc_harm(&s2);
	int ih;
	for (ih=0; ih<nh; ih++) {
		s2.data[ih] = s0->data[ih] * conj( s0->data[ih] );
	}
	
	/* Assign temporary CS and filter_freq arrays					*/
	struct cyclic_spectrum hpos;
	struct cyclic_spectrum hneg;
	cs_copy_parms(cs, &hpos);
	cs_copy_parms(cs, &hneg);
	cyclic_alloc_cs(&hpos);
	cyclic_alloc_cs(&hneg);
	
	struct filter_freq hf1;
	struct filter_freq hf_num;
	struct filter_freq hf_den;
	hf1.nchan = hf_num.nchan = hf_den.nchan = nc;
	filter_alloc_freq(&hf1);
	filter_alloc_freq(&hf_num);
	filter_alloc_freq(&hf_den);
	
	/* Initialise the estimate to 0 everywhere						*/
	int ic;
	for (ic=0; ic<nc; ic++) {
		hf1.data[ic] = 0.0 + I * 0.0;
	}
	
	/* Start the build at ic = rc, with H(rc) chosen to be real		*/
	fftwf_complex *csval = get_cs(cs,1,0,rc);
	fftwf_complex tmpval = *csval;
	tmpval /= s0->data[1];
	hf1.data[rc] = sqrt(sqrt(creal(tmpval*conj(tmpval)))) + I * 0.0;
	
	/* Iterate the estimator										*/
	int iter, itermax=nc/stepsize;
	for (iter=0; iter<itermax;iter++) {
		if (verbose) {
		printf("construct_filter_freq : iteration %d, %.5e\n",
			iter, creal(hf1.data[rc]));
		}
		/* Zero the numerator and denominator						*/
		for (ic=0; ic<nc; ic++) {
			hf_num.data[ic] = 0.0 + I * 0.0;
			hf_den.data[ic] = 0.0 + I * 0.0;
		}
						
		/* Populate cs filter arrays with the current estimate		*/
		filter2cs(&hpos, &hf1);
		filter2cs(&hneg, &hf1);
		/* Shear both +ve and -ve arrays by -alpha					*/
		shear = -1.0;
		cyclic_shear_cs(&hpos, shear, w);
		shear = 1.0;
		cyclic_shear_cs(&hneg, shear, w);
						
		/* Add up the contributions to numerator and denominator	*/	
		/* Negative alpha first										*/
		int ihmax, icmin, icmax;
		icmin = rc-2-iter*stepsize;
		icmax = rc  +iter/stepsize;
		if (icmin<0)    icmin = 0;
		if (icmax>nc-1) icmax = nc;
		for (ic=icmin; ic<icmax; ic++) {
			ihmax = harm_limit_cs_shear_plus(ic, cs);
			for (ih=1; ih<ihmax; ih++) {
				fftwf_complex *val1 = get_cs(&cs_neg,ih,0,ic);
				fftwf_complex *val2 = get_cs(&hneg,ih,0,ic);
				hf_num.data[ic] += (*val1) * (*val2);
				hf_den.data[ic] += (*val2) * conj(*val2) * s2.data[ih];
			}
		}
		/* Now positive alpha										*/
		icmin = rc+1-iter/stepsize;
		icmax = rc+3+iter*stepsize;
		if (icmin<0)    icmin = 0;
		if (icmax>nc-1) icmax = nc;		
		for (ic=icmin; ic<icmax; ic++) {
			ihmax = harm_limit_cs_shear_minus(ic, cs);
			for (ih=1; ih<ihmax; ih++) {
				fftwf_complex *val1 = get_cs(&cs_pos,ih,0,ic);
				fftwf_complex *val2 = get_cs(&hpos,ih,0,ic);
				hf_num.data[ic] += (*val1) * (*val2);
				hf_den.data[ic] += (*val2) * conj(*val2) * s2.data[ih];
			}
		}
		/* Now make a new estimate of hf1							*/
		for (ic=0; ic<nc; ic++) {
			if (creal(hf_den.data[ic])>0.0) {
				hf1.data[ic] += hf_num.data[ic] / hf_den.data[ic];
				/* Keep the update stable by averaging old and new	*/
				hf1.data[ic] /= 2.;
			}
		}
		/* Force the ic=rc filter value to be purely real			*/
		tmpval = hf1.data[rc];
		hf1.data[rc] += conj(tmpval);
		hf1.data[rc] /= 2.0;
		
	}
						
						
	/* Assign the constructed filter to the output array			*/
	for (ic=0;ic<nc;ic++) {
		hf->data[ic] = hf1.data[ic];
	}					
						
	/* Free the temporary arrays and return							*/
	filter_free_freq(&hf1);
	filter_free_freq(&hf_num);
	filter_free_freq(&hf_den);
	profile_free_harm(&s2);
	cyclic_free_cs(&cs_pos);
	cyclic_free_cs(&cs_neg);
	cyclic_free_cs(&hpos);
	cyclic_free_cs(&hneg);
						
	return(0);

}



