/* combine_profiles.c												*/
/*																	*/
/* M. Walker July 2011												*/
/*																	*/
/* Reads in pulse profiles, as a function of phase, adds them and	*/
/* outputs the result to file CombinedProfile.txt					*/
/*																	*/
/* Before reading each profile, the file is scanned to count the	*/
/* number of newline characters. Allocates array size equal to the	*/
/* number of newline characters in the first file. And checks that	*/
/* the number of newline characters is the same for all files		*/
/*																	*/
/* Needs cyclic_utils to actually read and write the profiles		*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <getopt.h>
#include <math.h>
#include <string.h>

#include "cyclic_utils.h"

void usage() {printf("combine_profiles file1 file2 [file3 ...]\n");}

int main(int argc, char *argv[]) {

	if (optind==argc) {
        usage();
        exit(1);
    }
	
	int i, nphase=0, lines=0, nread=0;
	struct profile_phase pp_tmp;
	struct profile_phase pp_combined;

	while (--argc>0) {
		/* There's at least one more profile to read				*/
		/* Scan each file to determine the number of lines			*/		
		FILE *f = fopen(*++argv, "r");
		char c;
		while ((c=getc(f))!=EOF) { if (c=='\n') { lines++; } }
		fclose(f);
		
		if (!nread) {
			nphase = lines; /* Define array size from 1st file		*/
			pp_tmp.nphase      = nphase;
			pp_combined.nphase = nphase;
			profile_alloc_phase(&pp_tmp);
			profile_alloc_phase(&pp_combined);
			for (i=0; i<nphase; i++) {
				pp_combined.data[i] = 0.0;
				pp_tmp.data[i]		= 0.0;
			}
		}
		else {					/* Check dimensions match 1st file	*/
			if (lines!=nphase) {
				printf("combine_profiles : incompatible dimensions\n");
				exit(1);
			}
		}
		read_profile(*argv, &pp_tmp);
		/* Add this to the total									*/
		for (i=0; i<nphase; i++) {
			pp_combined.data[i] += pp_tmp.data[i];
		}
		lines = 0;
		nread++;
	}
	
	/* Write out the result											*/
	const char *output_file = "CombinedProfile.txt";
	write_profile(output_file, &pp_combined);
	
	
	/* Free structs and return										*/
	profile_free_phase(&pp_tmp);
	profile_free_phase(&pp_combined);

    exit(0);
}