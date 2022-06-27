#----------------------------------------------------------------------------
#
# Program CHIMES
# "CHemical Interstellar Medium Evolution Solver"
#
# Jacques Le Bourlot and Evelyne Roueff - 2020
# Jacques.Lebourlot@obspm.fr
#
#----------------------------------------------------------------------------

   FC = gfortran
#  FFLAGS = -g3 -O0 -fno-second-underscore -Wall -Wextra -Wno-compare-reals -fcheck=all -fimplicit-none -std=f2008 -pedantic -ffpe-trap=zero,overflow -fbacktrace -gdwarf-2 -fall-intrinsics -Wno-unused-function
   #FFLAGS = -g3 -O0 -fno-second-underscore -Wall -Wextra -Wno-compare-reals -fcheck=all -fimplicit-none -std=f2008 -pedantic -fbacktrace -gdwarf-2 -fall-intrinsics -Wno-unused-function
   FFLAGS = -g -O3
   LIBS   = -llapack -lblas

#---------------------------------------------

#  FC = ifort
#  FFLAGS = -g3 -O0 -mkl -check all -warn all -check:noarg_temp_created -traceback -debug all -ftrapuv -fpe0 -diag-enable warn -fp-stack-check -mp1 -debug inline-debug-info
#  FFLAGS = -g -O2 -parallel -mkl -xSSSE3 -axSSSE3 -m64 -heap-arrays 1600

#---------------------------------------------

all: Chem_Evol Chem_Analys

DVODE_F90_M.o: DVODE_F90_M.f90
	$(FC) -c $(FFLAGS) DVODE_F90_M.f90

qeispack_Mod.o: qeispack_Mod.f90
	$(FC) -c $(FFLAGS) qeispack_Mod.f90

Chem_Aux.o: Chem_Aux.f90
	$(FC) -c $(FFLAGS) Chem_Aux.f90

Chem_Sub.o: Chem_Sub.f90 Chem_Aux.o
	$(FC) -c $(FFLAGS) Chem_Sub.f90

Chem_Evol: Chem_Evol.f90 DVODE_F90_M.o qeispack_Mod.o Chem_Sub.o Chem_Aux.o
	$(FC) $(FFLAGS) Chem_Aux.o Chem_Sub.o DVODE_F90_M.o qeispack_Mod.o Chem_Evol.f90 $(LIBS) -o Chem_Evol ; cp Chem_Evol ..

Chem_Analys: Chem_Analys.f90 Chem_Sub.o qeispack_Mod.o Chem_Aux.o
	$(FC) $(FFLAGS) Chem_Aux.o Chem_Sub.o qeispack_Mod.o Chem_Analys.f90 $(LIBS) -o Chem_Analys ; cp Chem_Analys ..

clean:
	\rm -r -f *.o *.mod *dat *graph *out Chem_Evol Chem_Evol_N Chem_Analys fort.9 *.dSYM *.gcno

