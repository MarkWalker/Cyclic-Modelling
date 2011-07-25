PROGS1 = filter_profile
PROGS2 = combine_profiles
CC = gcc
CFLAGS = -g -Wall -O3
LDLIBS = -lcfitsio -lfftw3f_threads -lfftw3f -lnlopt -lm
all: $(PROGS1) $(PROGS2)
clean:
	rm -rf *.o
install: all
	cp -f $(PROGS1) $(LOCAL)/bin;
	cp -f $(PROGS2) $(LOCAL)/bin;

filter_profile: filter_profile.o model_cyclic.o cyclic_utils.o 
combine_profiles: combine_profiles.o cyclic_utils.o 
