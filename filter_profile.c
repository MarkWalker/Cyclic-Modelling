/* filter_profile.c													*/
/*																	*/
/* Original code by P. Demorest, December 2010						*/
/* Substantially modified by M. Walker July 2011					*/
/*																	*/
/* Determine filter functions and profiles by least-squares fitting	*/
/* using the NLOPT optimization library.							*/
/*																	*/
/* Filter functions are optimised, with respect to a reference		*/
/* pulse profile, using NLOPT. Then a new profile function is		*/
/* generated directly by applying the filter functions to the data	*/
/*																	*/
/* Filter optimisation only available in lag space at present		*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <signal.h>
#include <getopt.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <fitsio.h>
#include <nlopt.h>
#include <string.h>

#include "cyclic_utils.h"
#include "model_cyclic.h"

#define fits_error_check_fatal() do { \
    if (status) { \
        fits_report_error(stderr, status); \
        exit(1); \
    } \
} while (0)

void usage() {printf("filter_extraction [Options] filename\n");
    printf("Options:\n");
    printf("  -i initialise (no optimisation)\n");
    printf("  -v verbose\n");
    printf("  -R fname  Use the Reference pulse profile in fname\n");
    printf("Must invoke with -i or -R options\n");
}

/* Functions for timing												*/
double cur_time_in_sec() {
    struct timeval tv;
    int rv=0;
    if ((rv=gettimeofday(&tv,NULL))!=0) { return(0); }
    return (tv.tv_sec+tv.tv_usec*1e-6);
}

/* Struct for passing the data and other information to nlopt		*/
struct cyclic_data {
    struct cyclic_spectrum *cs;	/* The data	themselves				*/
    struct profile_harm *s0;	/* Reference profile harmonics		*/
    struct cyclic_work *w;		/* FFTW plans, etc					*/
	int		*rc;				/* Freq channel : cimag(H(rc))=0	*/
};

/* Integers which can be accessed by various functions				*/
int verbose = 0;
int sample_ncalls = 0;

/* Return the sum of square differences between current model and	*/
/* data, using the functional form that nlopt wants.				*/
/* Compute the gradient of the sum-of-squares if needed.			*/
/* Vector "x" contains the parameter values H(freq).				*/
double cyclic_merit_nlopt(unsigned n, const double *x, 
        double *grad, void *_data) {

	extern int verbose;
	extern int sample_ncalls;
    static int ncalls=0;
    static double tot_time = 0.0;
    double t0 = cur_time_in_sec();
    ncalls++;
	sample_ncalls++;
	
    /* Pointer to input data										*/
    struct cyclic_data *data = (struct cyclic_data *)_data;

    /* check dimensions												*/
	if (n != 2*(data->w->nchan)-1) {
        fprintf(stderr, 
		"cyclic_merit_nlopt: error, inconsistent sizes!\n");
        exit(1);
    }
	
	/* Put the parameters x into the struct hf, for ease of use		*/
	int rchan = *(data->rc);
	struct filter_freq hf;
	hf.nchan = data->w->nchan;
    filter_alloc_freq(&hf);
	parms2struct(x, &hf, rchan);
		
    /* Compute the model cyclic spectrum from hf and s0				*/
	struct cyclic_spectrum cs_model;
	cs_copy_parms(data->cs, &cs_model);
	cyclic_alloc_cs(&cs_model);
	
	int imod = make_model_cs(&cs_model, &hf, data->s0, data->w);
	
    if (imod != 0) { fprintf(stderr, 
		"cyclic_merit_nlopt: error in make_model_cs (%d)\n",imod);
        exit(1);
    }	
	
    /* Evaluate and return the sum of |data-model|^2				*/
    double merit = cyclic_square_difference(data->cs, &cs_model);
	
		
	/* Evaluate the gradient, if needed								*/
    if (grad != NULL) {		
		struct filter_freq complex_gradient;
		complex_gradient.nchan=data->w->nchan;
		filter_alloc_freq(&complex_gradient);
		
		int mg;
		mg = merit_gradient(&complex_gradient, data->cs, &hf,
							data->s0, data->w);
				
		if (mg != 0) { fprintf(stderr, 
			"cyclic_merit_nlopt: error in merit_gradient (%d)\n",imod);
			exit(1);
		}
		
		/* Assign NLOPT-grad values from the complex_gradient		*/
		struct2parms(&complex_gradient, grad, rchan);
					
		filter_free_freq(&complex_gradient);
    }	
	
    /* Print timing info											*/
    double t1 = cur_time_in_sec();
    tot_time += t1-t0;
	if (verbose) {
		printf("Total ncalls=%d, avg %.2e sec/call, m = %.7e\n", 
			   ncalls, tot_time/(double)ncalls, merit);				
	}	
	
	cyclic_free_cs(&cs_model);
    filter_free_freq(&hf);

    return(merit);
}

/* Catch sigint														*/
int run=1;
void cc(int sig) { run=0; }

int main(int argc, char *argv[]) {

    int isub = 1, opt=0, opcheck=0, do_optimisation=0;
	extern int verbose;
	extern int sample_ncalls;
	char *ref_prof="";
    while ((opt=getopt(argc,argv,"ivR:"))!=-1) {
        switch (opt) {
            case 'i':
				opcheck++;
                break;
            case 'v':
                verbose++;
                break;
            case 'R':
                ref_prof = optarg;
				do_optimisation++;
				opcheck++;
                break;
        }
    }

    if (optind==argc) {
        usage();
        exit(1);
    }
    if (opcheck!=1) {
        usage();
        exit(1);
    }
	
	int ic, ih, j, is;
    int rv, status, nspec=1;
	
    /* Open fits datafile											*/
    fitsfile *f;
    fits_open_file(&f, argv[optind], READONLY, &status);		
    fits_error_check_fatal();

    /* Get basic dimensions											*/
    struct cyclic_work w;
    cyclic_load_params(f, &w, &nspec, &status);
    fits_error_check_fatal();
    
	if (verbose) {
       printf("Read nphase = %d, npol = %d, nchan = %d, nspec = %d\n", 
			   w.nphase, w.npol, w.nchan, nspec);
	   fflush(stdout);
    }
    
    int orig_npol = w.npol;
    w.npol = 1;

    /* Initialise FFTs												*/
    fftwf_init_threads();
    fftwf_plan_with_nthreads(2);
    if (verbose) { printf("Planning FFTs\n"); fflush(stdout); }
#define WF "/Users/mwalker/Software/C/cyclic_wisdom.dat"
    FILE *wf = fopen(WF,"r");
    if (wf!=NULL) { fftwf_import_wisdom_from_file(wf); fclose(wf); }
    rv = cyclic_init_ffts(&w);
    if (rv) {
        fprintf(stderr, "Error planning ffts (rv=%d)\n", rv);
        exit(1);
    }
    wf = fopen(WF,"w");
    if (wf!=NULL) { fftwf_export_wisdom_to_file(wf); fclose(wf); }

    /* Allocate some stuff											*/
    struct periodic_spectrum raw;
    struct cyclic_spectrum cs;
    struct filter_freq hf;
    struct profile_phase pp, pp_ref, pp_int;
    struct profile_harm  ph, ph_ref;

    raw.nphase = pp.nphase = pp_ref.nphase = pp_int.nphase = w.nphase;
    raw.nchan = cs.nchan = hf.nchan = w.nchan;
    cs.nharm = ph.nharm = ph_ref.nharm = w.nharm;
    raw.npol = orig_npol;
    cs.npol = 1;

    cyclic_alloc_ps(&raw);
    cyclic_alloc_cs(&cs);
    filter_alloc_freq(&hf);
    profile_alloc_phase(&pp);
    profile_alloc_phase(&pp_ref);
    profile_alloc_phase(&pp_int);
    profile_alloc_harm(&ph);
    profile_alloc_harm(&ph_ref);
	
	/* Initialise arrays for dynamic spectrum and optimised filters	*/
	float dynamic_spectrum[nspec][w.nchan];
	fftwf_complex optimised_filters[nspec][w.nchan];
	for (is=0; is<nspec; is++) {
		for (ic=0; ic<w.nchan;ic++) {
			dynamic_spectrum[is][ic]  = 0.;
			optimised_filters[is][ic] = 0. + I * 0.;
		}
	}
	
	/* Initialise arrays to record optimisation stats				*/
	float minima[nspec];
	int outcome[nspec];
	int	callno[nspec];
	for (is=0; is<nspec; is++) {
		minima[is]  = 0.0;
		outcome[is] = 0;
		callno[is]  = 0;
	}
	
	/* Initialise the intrinsic pulse profile to zero				*/
	for (j=0; j<pp_int.nphase; j++) { 
		pp_int.data[j] = 0.0;
	}
	
	if (do_optimisation) {
		/* Read in the reference pulse profile						*/
		read_profile(ref_prof, &pp_ref);
		/* Convert reference profile to harmonics					*/
		profile_phase2harm(&pp_ref, &ph_ref, &w);
		/* Ensure the zero-frequency term is zero					*/
		ph_ref.data[0] = 0.0 + I * 0.0;
	}
	
	
	/* Loop over all subintegrations								*/
	int noptimised=0;
	isub = 1;
	while (isub < nspec - 1) {
		
		if (verbose) {
			printf("Subintegration %d of %d\n", isub, nspec);
		}
		
		/* Load data												*/
		raw.npol = orig_npol;
		cyclic_load_ps(f, &raw, isub, &status);
		fits_error_check_fatal();
				
		/* Add polarisations without calibration					*/
		cyclic_pscrunch_ps(&raw, 1.0, 1.0);
		/* Only one polarisation from this point in loop			*/
		
		/* Convert input data to cyclic spectrum					*/
		cyclic_ps2cs(&raw, &cs, &w);
		
		/* The zero modulation-frequency component of the cyclic	*/
		/* spectrum gives us the dynamic spectrum -> output			*/
		int ic;
		for (ic=0; ic<w.nchan; ic++) {
			fftwf_complex *d1 = get_cs(&cs, 0, 0, ic);
			dynamic_spectrum[isub-1][ic]  = creal(*d1);
		}
		
		/* Initialise H(freq) to unity if this is the first sample	*/
		if (!noptimised) for (ic=0; ic<hf.nchan; ic++) {
			hf.data[ic] = 1.0 + I * 0.0;
		}
				
		/* Find ic : |cs(ic,1)| is largest							*/
		int rchan = maximum_cs1(&cs);
		if (verbose) {
			printf("Real channel = %d\n",rchan);
		}
		/* Rotate the phase of H so that H(rchan) is real			*/
		rotate_filter_phase(&hf, rchan);

		if (do_optimisation) {
			
			/* Normalise the reference profile : |H(freq)| ~ 1	*/
			normalise_profile(&ph_ref, &cs);

			if (!noptimised) {
				/* Build an approximate H(freq) by iteration		*/
				construct_filter_freq(&hf, &ph_ref, &cs, &w);
			}
			/* Enforce ph_ref as the reference profile				*/
			for (ih=0; ih<w.nharm; ih++) {
				ph.data[ih] = ph_ref.data[ih];
			}
			/* Fill in data struct for NLOPT						*/
			struct cyclic_data cdata;
			cdata.cs = &cs;
			cdata.s0 = &ph;
			cdata.w	 = &w;
			cdata.rc = &rchan;

			sample_ncalls = 0;
			/* Set up NLOPT minimizer								*/
			const int dim = 2*w.nchan-1; /* no. of free parameters	*/
			if (verbose) {
				printf("Number of fit parameters = %d\n", dim);
			}
			nlopt_opt op;
			op = nlopt_create(NLOPT_LD_LBFGS, dim);
			nlopt_set_min_objective(op, cyclic_merit_nlopt, &cdata);
			
			/* Assert the stopping criteria							*/
			double tolerance = 1.e-6;
			/* Fractional tolerance on the merit function			*/
			nlopt_set_ftol_rel(op, tolerance);
			if (verbose) {
				printf("NLOPT: set FTOL-rel = %.5e\n",tolerance);
			}
			tolerance = 1.e-3;
			/* Fractional tolerance on the parameter values			*/
			nlopt_set_xtol_rel(op, tolerance);
			if (verbose) {
			 printf("NLOPT: set XTOL-rel = %.5e\n",tolerance);
			}
			
			
			/* Set initial values of parameters	to current H(freq)	*/
			double *x = (double *)malloc(sizeof(double) * dim);			
			
			if (x==NULL) {
				fprintf(stderr, "malloc(x) : insufficient space\n");
				exit(1);
			}
			struct2parms(&hf, x, rchan);			
			
			/* Run NLOPT optimization								*/
			double min;
			outcome[isub-1] = nlopt_optimize(op, x, &min);
			minima[isub-1]  = min;
			callno[isub-1]  = sample_ncalls;

			/* Construct H(freq) from the optimum values of "x"		*/
			parms2struct(x, &hf, rchan);
						
			/* Free-up allocations									*/
			free(x);
			nlopt_destroy(op);
		}
		
		/* Put current optimised filter(freq) into array			*/
		for (ic=0; ic<hf.nchan; ic++) {
			optimised_filters[isub-1][ic]=hf.data[ic];
		}
		
		/* Get optimised profile, given filter,	for this sub-int	*/
		int iprof = 0;
		iprof = optimise_profile(&ph, &cs, &hf, &w);
		if (iprof!=0) {
			fprintf(stderr, "Error in optimise_profile.\n");
			exit(1);
		}
		
        /* Convert profile(harmonic) to profile(phase)				*/
        ph.data[0] = 0.0 + I * 0.0;
        profile_harm2phase(&ph, &pp, &w);
				
		/* Add profile for current subint to the intrinsic estimate */
		for (j=0; j<pp_int.nphase; j++) { 
			pp_int.data[j] += pp.data[j];
		}
					
		isub++;
		noptimised++;
	}

	char *inputptr = argv[optind];
	char dotchar = '.';
	char *dotpos = strrchr(inputptr,dotchar);
	size_t fnamelength = dotpos - inputptr + 1 ;
	char *outputptr = inputptr;	
	strcpy(outputptr,inputptr);
		
	/* Output intrinsic profile (= scattered profile if opt = -i)	*/
	outputptr[fnamelength] = '\0';
	strcat(outputptr,"profile.txt");
	write_profile(outputptr, &pp_int);
		
	/* Output the optimised filters									*/
	outputptr[fnamelength] = '\0';
	strcat(outputptr,"filters.txt");
	FILE *fpointer = fopen(outputptr, "w");
	fprintf(fpointer, "%d\n", nspec);
	fprintf(fpointer, "%d\n", w.nchan);
	for (is=0; is<nspec; is++) {
		for (ic=0; ic<w.nchan; ic++) {
			fprintf(fpointer, "%.7e %.7e\n",
					(float)creal(optimised_filters[is][ic]),
					(float)cimag(optimised_filters[is][ic]));
		}
	}
    fprintf(fpointer,"\n\n");
    fclose(fpointer);
	
	/* Output the dynamic spectrum									*/
	outputptr[fnamelength] = '\0';
	strcat(outputptr,"dynamic_spectrum.txt");
	fpointer = fopen(outputptr, "w");
	fprintf(fpointer, "%d\n", nspec);
	fprintf(fpointer, "%d\n", w.nchan);
	for (is=0; is<nspec; is++) {
		for (ic=0; ic<w.nchan; ic++) {
			fprintf(fpointer, "%.7e\n", dynamic_spectrum[is][ic]);
		}
	}
    fprintf(fpointer,"\n\n");
    fclose(fpointer);
	
	/* Output the optimisation stats								*/
	outputptr[fnamelength] = '\0';
	strcat(outputptr,"optimisation_stats.txt");
	fpointer = fopen(outputptr, "w");
	fprintf(fpointer, "%d\n", nspec);
	for (is=0; is<nspec; is++) {
		fprintf(fpointer, "%d %d %d %.7e\n", is+1, outcome[is],
				callno[is], minima[is]);
	}
    fprintf(fpointer,"\n\n");
    fclose(fpointer);
	
	/* Free-up the structs											*/
    cyclic_free_ps(&raw);
	cyclic_free_cs(&cs);
    filter_free_freq(&hf);
    profile_free_phase(&pp);
    profile_free_harm(&ph);
    profile_free_phase(&pp_ref);
    profile_free_harm(&ph_ref);
    profile_free_phase(&pp_int);
	cyclic_free_ffts(&w);
	
    exit(0);
}