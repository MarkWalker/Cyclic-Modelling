/* cyclic_utils.h													*/

/* Basic structs/functions to manipulate cyclic spectra				*/

#ifndef _CYCLIC_UTILS_H
#define _CYCLIC_UTILS_H

#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <fitsio.h>

/* Conventions:
 *   nlag = nchan, and are arranged as
 *      0,1,..,nlag/2,-(nlag/2-1),..,-1
 *   Spectra always have nchan points at both pos and neg freqs
 *   Harmonic arrays only have nharm positive components, and include DC
 */

/* Periodic spectrum, with axes phase, pol, chan */
struct periodic_spectrum {
    int nphase;
    int npol;
    int nchan;
    int imjd;
    double fmjd;
    double ref_phase;
    double ref_freq;
    double rf;
    double bw;
    float *data;
};

/* Cyclic spectrum */
struct cyclic_spectrum {
    int nharm;
    int npol;
    int nchan;
    int imjd;
    double fmjd;
    double ref_phase;
    double ref_freq;
    double rf;
    double bw;
    fftwf_complex *data;
};

/* Cyclic correlation */
struct cyclic_correlation {
    int nharm;
    int npol;
    int nlag;
    int imjd;
    double fmjd;
    double ref_phase;
    double ref_freq;
    double rf;
    double bw;
    fftwf_complex *data;
};

/* Periodic correlation */
struct periodic_correlation {
    int nphase;
    int npol;
    int nlag;
    int imjd;
    double fmjd;
    double ref_phase;
    double ref_freq;
    double rf;
    double bw;
    fftwf_complex *data;
};

/* Filter functions in time/freq domain */
struct filter_time {
    int nlag;
    fftwf_complex *data;
};
struct filter_freq {
    int nchan;
    fftwf_complex *data;
};

/* Pulse profiles in phase/harmonic domain */
struct profile_phase {
    int nphase;
    float *data;
};
struct profile_harm {
    int nharm;
    fftwf_complex *data;
};

/* Struct to hold working info (fftw plans, etc) */
struct cyclic_work {

    int npol;

    int nchan;
    int nlag;

    int nphase;
    int nharm;

    /* Add to these as needed */
    fftwf_plan ps2cs;
    fftwf_plan cs2cc;
    fftwf_plan cc2cs;

    fftwf_plan time2freq;
    fftwf_plan freq2time;
    fftwf_plan phase2harm;
    fftwf_plan harm2phase;
};

/* Help save lots of characters... */
typedef struct periodic_spectrum PS;
typedef struct cyclic_spectrum CS;
typedef struct cyclic_correlation CC;
typedef struct periodic_correlation PC;

/* Simple get data funcs to avoid indexing problems, no bounds check for now */
static inline float *get_ps(PS *d, int iphase, int ipol, int ichan) {
    return &d->data[iphase + d->nphase*ipol + d->nphase*d->npol*ichan];
}
static inline fftwf_complex *get_cs(CS *d, int iharm, int ipol, int ichan) {
    return &d->data[iharm + d->nharm*ipol + d->nharm*d->npol*ichan];
}
static inline fftwf_complex *get_cc(CC *d, int iharm, int ipol, int ilag) {
    return &d->data[iharm + d->nharm*ipol + d->nharm*d->npol*ilag];
}
static inline fftwf_complex *get_pc(PC *d, int iphase, int ipol, int ilag) {
    return &d->data[iphase + d->nphase*ipol + d->nphase*d->npol*ilag];
}

/* Alloc/free datatypes */
void cyclic_alloc_ps(PS *d);
void cyclic_alloc_cs(CS *d);
void cyclic_alloc_cc(CC *d);
void cyclic_alloc_pc(PC *d);
void cyclic_free_ps(PS *d);
void cyclic_free_cs(CS *d);
void cyclic_free_cc(CC *d);
void cyclic_free_pc(PC *d);
void filter_alloc_time(struct filter_time *f);
void filter_alloc_freq(struct filter_freq *f);
void filter_free_time(struct filter_time *f);
void filter_free_freq(struct filter_freq *f);
void profile_alloc_phase(struct profile_phase *p);
void profile_alloc_harm(struct profile_harm *p);
void profile_free_phase(struct profile_phase *p);
void profile_free_harm(struct profile_harm *p);

/* Load dimension params from datafile */
/* MAW modified to include *nspec */
int cyclic_load_params(fitsfile *f, struct cyclic_work *w,
					   int *nspec, int *status);

/* Load one periodic spectrum from datafile */
int cyclic_load_ps(fitsfile *f, PS *d, int idx, int *status);

/* Init fft plans for datatype conversion */
int cyclic_init_ffts(struct cyclic_work *w);
void cyclic_free_ffts(struct cyclic_work *w);

/* Add polarizations in-place to get total intensity, allowing		*/
/* x/y gain factors to be applied.									*/
/* MAW 7/7/11 : fixed polarisation interleave problem				*/
int cyclic_pscrunch_ps(PS *d, float xgain, float ygain);

/* Sum over freq in periodic spectrum */
int cyclic_fscrunch_ps(struct profile_phase *out, PS *in);

/* Added by MAW 17/7/2011											*/
/* Sums the input cyclic spectrum over radio frequency				*/
/* and puts the result into the output harmonic profile				*/
int cyclic_fscrunch_cs(struct profile_harm *out, const CS *in);
	
/* Conversion routines */
void cyclic_ps2cs(PS *in, CS *out, const struct cyclic_work *w);
void cyclic_cs2cc(CS *in, CC *out, const struct cyclic_work *w);
void cyclic_cc2cs(CC *in, CS *out, const struct cyclic_work *w);
void filter_time2freq(struct filter_time *in,
					  struct filter_freq *out,
					  const struct cyclic_work *w);
void filter_freq2time(struct filter_freq *in,
					  struct filter_time *out,
					  const struct cyclic_work *w);
void profile_phase2harm(struct profile_phase *in,
						struct profile_harm *out,
						const struct cyclic_work *w);
void profile_harm2phase(struct profile_harm *in, 
						struct profile_phase *out,
						const struct cyclic_work *w);

/* Added by MAW 16/7/2011. Modelled on PMD's cyclic_shift_cs		*/
int cyclic_shear_cs(CS *d, double shear, const struct cyclic_work *w);
	
/* Mean square difference functions for the various structs			*/
double profile_ms_difference(struct profile_harm *p1,
							 struct profile_harm *p2, int max_harm);
double filter_ms_difference(struct filter_time *f1,
							struct filter_time *f2);

/* MAW 14/7/2011. Modified version of PMD's	cyclic_ms_difference	*/
double cyclic_square_difference(const CS *cs1, const CS *cs2);

/* Output simple text-based versions of various quantities			*/
void write_profile(const char *fname, struct profile_phase *p);
void write_fprofile(const char *fname, struct profile_harm *p);
void write_filter(const char *fname, struct filter_time *h);
void write_filter_freq(const char *fname, struct filter_freq *h);

/* Read in the pulse profile										*/
void read_profile(const char *fname, struct profile_phase *pp);

/* Finds (roughly) where the filter function is maximum	by			*/
/* looking at the absolute square value of the cyclic spectrum		*/
/* at unit modulation freq											*/
/* Added by MAW 13/07/2011											*/
int	maximum_cs1(const CS *cs);

/* Finds the root-mean-square of the cyclic spectrum at harmonic	*/
/* number = iharm. Added by MAW 14/07/2011											*/
float rms_cs(const CS *cs, const int iharm);

/* Copies parameters from cs1 struct into cs2 struct				*/
/* Added by MAW 16/07/2011											*/
int	cs_copy_parms(const CS *cs1, CS *cs2);

/* Copies data from cs1 into cs2									*/
/* Added by MAW 16/07/2011											*/
int	cs_copy_data(const CS *cs1, CS *cs2);

/* Subtracts cs1 from cs2, placing the result in cs2				*/
/* Added by MAW 15/07/2011											*/
int	cs_subtract(const CS *cs1, CS *cs2);

/* Multiplies cs1 by cs2, placing the result in cs2					*/
/* Added by MAW 15/07/2011											*/
int	cs_multiply(const CS *cs1, CS *cs2);

/* Forms the conjugate of cs1 (in place)							*/
/* Added by MAW 15/07/2011											*/
int	cs_conjugate(CS *cs1);

/* Fills the cyclic spectrum cs1 with copies of the filter hf		*/
/* Added by MAW 15/07/2011											*/
int	filter2cs(CS *cs1, const struct filter_freq *hf);

/* Routine added by MAW 15/07/2011									*/
/* Fills the cyclic spectrum cs1 uniformly with the profile s0		*/
int	profile2cs(CS *cs1, const struct profile_harm *s0);
	
/* Rotates the phase of H(freq) so that H(rc) is purely real		*/
/* Added by MAW 14/07/2011											*/
int rotate_filter_phase(struct filter_freq *hf, const int rc);

/* Puts the 2*nchan-1 parameters x into the nchan complex numbers	*/
/* making up hf, with cimag(hf(rchan))=0							*/
/* Added by MAW 14/07/2011											*/
int parms2struct(const double *x, struct filter_freq *hf, 
				 const int rchan);

/* Takes the 2*nchan-1 parameters x from the nchan complex			*/
/* numbers making up hf. Omits Im(hf(rchan)) which is zero			*/
/* Added by MAW 14/07/2011											*/
int struct2parms(const struct filter_freq *hf, double *x, 
				 const int rchan);


/* Returns the min and max channel numbers giving a valid CS		*/
/* estimate at harmonic # iharm										*/
int	chan_limits_cs(int *chan_min, int *chan_max, const int iharm, 
				   const CS *cs);

/* Determine the maximum abs(harmonic no.) to which the CS is valid	*/
int	harm_limit_cs(const int ic, const CS *cs);

/* Returns the maximum (absolute value of the) harmonic number		*/
/* giving a valid sheared CS estimate. Use this limit for case		*/
/* where the shear is < 0											*/
int	harm_limit_cs_shear_minus(const int ichan, const CS *cs);

/* Returns the maximum (absolute value of the) harmonic number		*/
/* giving a valid sheared CS estimate. Use this limit for case		*/
/* where the shear is > 0											*/
int	harm_limit_cs_shear_plus(const int ichan, const CS *cs);

/* Normalises the reference pulse profile harmonics so that			*/
/* the implied filter amplitudes are rms(|H(freq)|) ~ 1				*/
/* Function added by MAW 22/07/2011									*/
int normalise_profile(struct profile_harm *s0, const CS *cs);


#endif
