#
# Makefile for EBM
#
FC =/usr/local/bin/ifort
FCFLAG = -r8 -align all -static-intel 
#FLAGS	= -r8 -align all -static-intel
TARGET	= driver
OBJECTS = driver.o
WDIR	= /Users/irenebonati/Desktop/PhD/Projects/CO2_Condensation/Ramses_Code

.PHONY: all test clean

$(TARGET) : $(OBJECTS)
	$(FC) $(FLAGS) -o $(TARGET) $(OBJECTS)

driver.o : $(WDIR)/driver.f90
	$(FC) $(FLAGS) -c $(WDIR)/driver.f90

clean:
	rm -rf $(OBJECTS) $(TARGET)

new:
	$(MAKE) clean
	$(MAKE)
