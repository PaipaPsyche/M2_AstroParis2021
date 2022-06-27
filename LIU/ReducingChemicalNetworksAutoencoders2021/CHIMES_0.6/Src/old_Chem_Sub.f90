! Project CHIMES
! Upgrade: JLB - June to November 2020
! Physics related Routines

MODULE Chem_Sub

   USE Chem_Aux
   USE qeispack

   !---constant gama :       5/3 = 1.6667 (monoatomic gas), 7/5 = 1.4000 (diatomic gas)
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: gama   = 1.40_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: xmh    = 1.67353283804963e-24_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: xmp    = 1.67262192369e-24_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: xme    = 9.1093837015e-28_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: xqe    = 4.80320471257026e-10_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: xk     = 1.380649e-16_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: xr     = 8.314462618e7_dp 
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: everg  = 1.602176634e-12_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: amu    = 1.66053906660e-24_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: calev  = 0.0433641042418009_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: Myr    = 31556925216000.0_dp
   REAL (KIND=dp)                     , PARAMETER, PUBLIC :: istiny = 1.0e-60_dp

   ! Dimensions
   INTEGER                            , PARAMETER, PUBLIC :: natom   = 21          ! Number of different atoms (depletions)
   INTEGER                            , PARAMETER, PUBLIC :: npart   = 7           ! Number of "pseudo-species"
   INTEGER                            , PARAMETER, PUBLIC :: nmhd    = 2           ! Number of MHD equations (usually 2) 
   INTEGER                                       , PUBLIC :: nspec                 ! Number of chemical species including electr
   INTEGER                                       , PUBLIC :: nx1, nx2              ! Index of chemical evolution equations
   INTEGER                                       , PUBLIC :: neqev                 ! Total number of evolution equations = nmhd + nspec
   INTEGER                                       , PUBLIC :: nn, ni, nneg          ! Number of neutral, ionized and negatively charged species
   INTEGER                                       , PUBLIC :: neqt                  ! Number of chemical equations
   ! For accurate spectral resolution, use mfft = 14. mfft = 12 is enough for most applications.
   !INTEGER                           , PARAMETER, PUBLIC :: mfft = 14             ! Size of vector data used for FFT
   !INTEGER                           , PARAMETER, PUBLIC :: nfft = 16384          ! nfft = 2**mfft
   INTEGER                            , PARAMETER, PUBLIC :: mfft = 12             ! Size of vector data used for FFT
   INTEGER                            , PARAMETER, PUBLIC :: nfft = 4096           ! nfft = 2**mfft

   ! I/O related variables
   CHARACTER (LEN=200)                           , PUBLIC :: sh_cmd                ! Shell command
   CHARACTER (LEN=100)                           , PUBLIC :: outdir                ! Name of Output Directory
   CHARACTER (LEN=50)                            , PUBLIC :: rootnm                ! Root name of input and output files (read on command line)
   CHARACTER (LEN=50)                            , PUBLIC :: fichi                 ! File names and I/O units
   INTEGER                            , PARAMETER, PUBLIC :: iscrn = 6             ! Write on screen
   INTEGER                            , PARAMETER, PUBLIC :: iread = 10            ! Read file
   INTEGER                            , PARAMETER, PUBLIC :: iwrtc = 11            ! Write chemistry analysis file
   INTEGER                            , PARAMETER, PUBLIC :: iwrto = 12            ! Write to full output file (is it really useful?)
   INTEGER                            , PARAMETER, PUBLIC :: iwrtm = 13            ! Write all maxima of H
   INTEGER                            , PARAMETER, PUBLIC :: iwrtf = 14            ! Write Fourier transform file
   INTEGER                            , PARAMETER, PUBLIC :: iwrtg = 15            ! Write in columns for plotting
   INTEGER                            , PARAMETER, PUBLIC :: iwrtz = 17            ! Write final state and extrema

   ! Input parameters
   ! F_St_Eq = 1 : T and nH constant
   ! F_St_Eq = 2 : isochoric (thermal balance + nH constant)
   ! F_St_Eq = 3 : isobaric
   INTEGER                                       , PUBLIC :: F_St_Eq               ! Physical case
   REAL (KIND=dp)                                , PUBLIC :: densh                 ! Proton density (i.e.n_H)
   REAL (KIND=dp)                                , PUBLIC :: zeta                  ! Cosmic rays ionization rate
   REAL (KIND=dp)                                , PUBLIC :: rad, av               ! Radiative field
   REAL (KIND=dp)                                , PUBLIC :: rop                   ! Ortho to Para ratio
   REAL (KIND=dp)                                , PUBLIC :: radgr                 ! Grain radius
   REAL (KIND=dp)                                , PUBLIC :: rhogr                 ! Bulk grain volumic density
   REAL (KIND=dp)                                , PUBLIC :: ratiog                ! Mass Gr / Mass gaz
   REAL (KIND=dp)                                , PUBLIC :: tgr                   ! Grain temperature
   REAL (KIND=dp)    , DIMENSION (:), ALLOCATABLE, PUBLIC :: abin                  ! Species initial abundances
   REAL (KIND=dp)                                , PUBLIC :: h0                    ! Maximal time step
   REAL (KIND=dp)                                , PUBLIC :: tend                  ! Total integration time
   INTEGER                                       , PUBLIC :: ks    = 1             ! Write output every ks steps
   INTEGER                                       , PUBLIC :: F_FFT = 0             ! Flag Fourier transform of H(t)
   INTEGER                                       , PUBLIC :: F_CH_A = 0            ! Flag Chemistry analysis
   REAL (KIND=dp)                                , PUBLIC :: csco                  ! Scaling of C+, O and CO
   REAL (KIND=dp)                                , PUBLIC :: frnH                  ! Scaling ratio (nH and zeta)
   REAL (KIND=dp)                                , PUBLIC :: poc1, poc2            ! C+O and C/O depletion parameters

   !---conserved species - identical to PDR code
   CHARACTER (LEN=8)  , DIMENSION (natom+npart+1), PUBLIC :: sp
   DATA sp / 'h       ', 'c       ', 'n       ', 'o       ', &                     ! Identical to the PDR code
             'he      ', 'd       ', 'c*      ', 'n*      ', &
             'o*      ', 'ar      ', 'f       ', 'na      ', &
             'mg      ', 'gr      ', 'si      ', 'ne      ', &                     ! "gr" used in place of "al"
             's       ', 'cl      ', 'ca      ', 'fe      ', &
             'electr  ', 'photon  ', 'crp     ', 'phosec  ', &
             'grain   ', '2h      ', 'xray    ', 'elecsec ', &
             '        ' /

   !--- Specific indexes used for various purposes in the code
   INTEGER                                       , PUBLIC :: i_h    = 0
   INTEGER                                       , PUBLIC :: i_c    = 0
   INTEGER                                       , PUBLIC :: i_n    = 0
   INTEGER                                       , PUBLIC :: i_o    = 0
   INTEGER                                       , PUBLIC :: i_he   = 0
   INTEGER                                       , PUBLIC :: i_d    = 0
   INTEGER                                       , PUBLIC :: i_c13  = 0
   INTEGER                                       , PUBLIC :: i_n15  = 0
   INTEGER                                       , PUBLIC :: i_o18  = 0
   INTEGER                                       , PUBLIC :: i_ar   = 0
   INTEGER                                       , PUBLIC :: i_f    = 0
   INTEGER                                       , PUBLIC :: i_na   = 0
   INTEGER                                       , PUBLIC :: i_mg   = 0
   INTEGER                                       , PUBLIC :: i_gr   = 0                 ! Used to be i_al
   INTEGER                                       , PUBLIC :: i_si   = 0
   INTEGER                                       , PUBLIC :: i_ne   = 0
   INTEGER                                       , PUBLIC :: i_s    = 0
   INTEGER                                       , PUBLIC :: i_cl   = 0
   INTEGER                                       , PUBLIC :: i_ca   = 0
   INTEGER                                       , PUBLIC :: i_fe   = 0
   INTEGER                                       , PUBLIC :: i_h2   = 0
   INTEGER                                       , PUBLIC :: i_hd   = 0
   INTEGER                                       , PUBLIC :: i_c2   = 0
   INTEGER                                       , PUBLIC :: i_n2   = 0
   INTEGER                                       , PUBLIC :: i_h2o  = 0
   INTEGER                                       , PUBLIC :: i_oh   = 0
   INTEGER                                       , PUBLIC :: i_ch   = 0
   INTEGER                                       , PUBLIC :: i_ch4  = 0
   INTEGER                                       , PUBLIC :: i_cd5p = 0
   INTEGER                                       , PUBLIC :: i_nh3  = 0
   INTEGER                                       , PUBLIC :: i_sh   = 0
   INTEGER                                       , PUBLIC :: i_co   = 0
   INTEGER                                       , PUBLIC :: i_hp   = 0
   INTEGER                                       , PUBLIC :: i_cp   = 0
   INTEGER                                       , PUBLIC :: i_sp   = 0
   INTEGER                                       , PUBLIC :: i_np   = 0
   INTEGER                                       , PUBLIC :: i_nhp  = 0
   INTEGER                                       , PUBLIC :: i_fep  = 0
   INTEGER                                       , PUBLIC :: i_h2dp = 0
   INTEGER                                       , PUBLIC :: i_c13p = 0
   INTEGER                                       , PUBLIC :: i_n15p = 0

   INTEGER       , DIMENSION (natom-1)           , PUBLIC :: iconv               ! Species used in conservation equations
   INTEGER       , DIMENSION (:)    , ALLOCATABLE, PUBLIC :: lconv               ! Flag for these species

   !--- Chemical reactions parameters
   INTEGER       , DIMENSION (:)    , ALLOCATABLE, PUBLIC :: itype               ! Type of reactions (as in PDR code)
   REAL (KIND=dp), DIMENSION (:)    , ALLOCATABLE, PUBLIC :: gamm, alpha, bet    ! Reaction rates (Arhenius coefficients)
   REAL (KIND=dp), DIMENSION (:)    , ALLOCATABLE, PUBLIC :: xkrate              ! Reaction rates (computed)
   REAL (KIND=dp), DIMENSION (:)    , ALLOCATABLE, PUBLIC :: de                  ! Endo/exothermicity
   INTEGER       , DIMENSION (:,:)  , ALLOCATABLE, PUBLIC :: react               ! Index of species involved in chemical reactions
   REAL (KIND=dp), DIMENSION (:)    , ALLOCATABLE, PUBLIC :: summ
   REAL (KIND=dp), DIMENSION (:)    , ALLOCATABLE, PUBLIC :: charg
   REAL (KIND=dp), DIMENSION (:)    , ALLOCATABLE, PUBLIC :: xmol, enth
   INTEGER       , DIMENSION (:,:)  , ALLOCATABLE, PUBLIC :: ael
   REAL (KIND=dp), DIMENSION (natom)             , PUBLIC :: depl, depl0
   CHARACTER (LEN=23), DIMENSION (:), ALLOCATABLE, PUBLIC :: observ
   CHARACTER (LEN=8) , DIMENSION (:), ALLOCATABLE, PUBLIC :: speci

   !--- H2 excitation variable and thermal balance
   INTEGER                        , PARAMETER, PUBLIC :: n_rot = 20
   REAL (KIND=dp), DIMENSION (n_rot)         , PUBLIC :: abj
   REAL (KIND=dp)                            , PUBLIC :: bthb
   REAL (KIND=dp)                            , PUBLIC :: ynn
   REAL (KIND=dp)                            , PUBLIC :: bnn
   REAL (KIND=dp)                            , PUBLIC :: bee
   REAL (KIND=dp)                            , PUBLIC :: bchim
   REAL (KIND=dp)                            , PUBLIC :: gamh2
   REAL (KIND=dp)                            , PUBLIC :: gamgr
   REAL (KIND=dp)                            , PUBLIC :: gampeg
   REAL (KIND=dp)                            , PUBLIC :: zlambd
   REAL (KIND=dp)                            , PUBLIC :: xlamoh
   REAL (KIND=dp)                            , PUBLIC :: xlamco
   REAL (KIND=dp)                            , PUBLIC :: xlah2o
   REAL (KIND=dp)                            , PUBLIC :: xlamc
   REAL (KIND=dp)                            , PUBLIC :: xlamcp
   REAL (KIND=dp)                            , PUBLIC :: xlamo
   REAL (KIND=dp)                            , PUBLIC :: xlambn
   REAL (KIND=dp)                            , PUBLIC :: wthb
   REAL (KIND=dp)                            , PUBLIC :: xlambe
   REAL (KIND=dp)                            , PUBLIC :: gamrc

   !--- Integration variables and auxiliaries
   REAL (KIND=dp)                            , PUBLIC :: hs, ts              ! Times in Myr
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE, PUBLIC :: fy                  ! Auxiliary variable for evolution equations
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE, PUBLIC :: f0                  ! 
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE, PUBLIC :: abrelh2             ! 
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE, PUBLIC :: abev                ! Species abundances thru evolution
   REAL (KIND=dp)                            , PUBLIC :: tempr               ! Temperature
   REAL (KIND=dp)                            , PUBLIC :: press               ! Pression
   REAL (KIND=dp)                            , PUBLIC :: xdens               ! Number density (Sum of abundances)
   REAL (KIND=dp)                            , PUBLIC :: para, orth          ! Para and ortho fractions
   REAL (KIND=dp)                            , PUBLIC :: xnh, xnh2, xno      ! Useful shortcuts for inner computations
   REAL (KIND=dp)                            , PUBLIC :: xnc, xncp, xnh2o
   REAL (KIND=dp)                            , PUBLIC :: xnoh, xnco, xnhe
   REAL (KIND=dp)                            , PUBLIC :: xnhpl
   REAL (KIND=dp)                            , PUBLIC :: xngr                ! Dust abundance
   REAL (KIND=dp)                            , PUBLIC :: t_trans             ! Estimated transitory time
   REAL (KIND=dp)                            , PUBLIC :: t_per               ! Estimated period - Hand set for now
   REAL (KIND=dp)                            , PUBLIC :: t_sav               ! Time for next output
   INTEGER                                   , PUBLIC :: i_sav               ! index of next output
   REAL (KIND=dp), DIMENSION (3)             , PUBLIC :: tcur
   REAL (KIND=dp), DIMENSION (3)             , PUBLIC :: tecur
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE, PUBLIC :: abc1, abc2, abc3
   REAL (KIND=dp),                             PUBLIC :: ttemin, ttemax
   REAL (KIND=dp),                             PUBLIC :: temin, temax
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE, PUBLIC :: abmin, abmax
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE, PUBLIC :: ttmin, ttmax
   INTEGER                                   , PUBLIC :: idbg

   ! Flags
   INTEGER                                   , PUBLIC :: F_Jac = 0           ! Flag Jacobian Matrix (1: Compute)
   INTEGER, PARAMETER                        , PUBLIC :: F_EIGEN = 0         ! Flag Eigenvalues (1: Compute)

   PRIVATE

   PUBLIC :: init, lectur, diffun, steady_st, max_ht, extrema

CONTAINS

!**************
SUBROUTINE init
!**************
!---------------------------------------------------------------------
!  Input parameters
!---------------------------------------------------------------------

   USE Chem_Aux

   IMPLICIT NONE

   CHARACTER (LEN=9)  :: ttag
   LOGICAL            :: is_there

   CALL get_command_argument(1, rootnm)
   IF (LEN_TRIM(rootnm) == 0) rootnm = "Base"

   !--- Read model parameters
   fichi = "Data/" // TRIM(ADJUSTL(rootnm)) // "_Param.dat"
   INQUIRE (FILE=fichi, EXIST=is_there)
   IF (is_there .EQV. .FALSE.) THEN
      PRINT *, "# File ", fichi, " does not exists"
      PRINT *, "# Please provide a valid root name"
      STOP
   ENDIF

   OPEN  (iread, FILE = fichi, STATUS='old')
   READ  (iread,*) F_St_Eq                      ! Specify state equation
   READ  (iread,*) densh                        ! Initial density
   READ  (iread,*) tempr                        ! Initial temperature
   READ  (iread,*) zeta                         ! Cosmic rays ionization rate
   READ  (iread,*) rad                          ! External radiation field
   READ  (iread,*) av                           ! Position in cloud (Av)
   READ  (iread,*) rop                          ! Ortho / para ratio
   IF (rop > 1.0_dp) THEN
      PRINT *, " rop =", rop, " is not possible"
      PRINT *, " Please modify this value and rerun"
      STOP
   ENDIF
   READ  (iread,*) tgr                          ! Dust temperature
   READ  (iread,*) ratiog                       ! Dust to gas mass ratio
   READ  (iread,*) radgr                        ! Dust radius
   READ  (iread,*) rhogr                        ! Dust mass density
   READ  (iread,*) h0                           ! Time step (Myr)
   READ  (iread,*) tend                         ! Total integration time (Myr)
   READ  (iread,*)
   READ  (iread,*) t_per                        ! Estimated period of oscillations
   READ  (iread,*) ks                           ! Output decimation
   READ  (iread,*) F_FFT                        ! FFT Flag
   READ  (iread,*) F_CH_A                       ! Chemical Analysis Flag

   ! Optional parameters used for specific tasks
   READ  (iread,*) csco                         ! C / O specific setting
   READ  (iread,*) frnH                         ! zeta and nH scaling ratio
   READ  (iread,*) poc1                         ! C and O depletion scaling
   READ  (iread,*) poc2                         !        ---
   CLOSE (iread)

   ! Create output directory if needed
   outdir = "Out/" // TRIM(ADJUSTL(rootnm)) // '/'
   sh_cmd = 'sh -c "if [ ! -d ' // TRIM(outdir) // " ]; then mkdir " // TRIM(outdir) // ' ; fi "'
   CALL EXECUTE_COMMAND_LINE(sh_cmd)

   !--- If we want to analyse the chemistry, then somme data need to be tuned to the specific case
   ! t_per should be estimated by the user from a previous run.
   IF (F_CH_A == 1) THEN
      t_sav = tend - 1.2_dp * t_per
      i_sav = 0
   ENDIF

   !--- Set transitory time to 1/4 of tend
   !--- Results earlier than t_trans are discarded for an FFT computation
   !--- Note: This must be ajusted when close to a bifurcation point
   t_trans = 0.25_dp * tend
   !--- Convert h0 and tend from Myr to s
   ts      = 0.0_dp
   h0      = h0 * Myr
   tend    = tend * Myr

   ! The following specific cases have been used to explore parameter spae
   ! During the work on Roueff & le Bourlot, A&A, 2020
   ! They are not inended to be used by default.
   ! Adapt to your needs

   ! Ajust nH and zeta at constant I_ep
   ! Useful to explore isochore models
   IF (frnH > 0.0_dp) THEN
      densh = densh * frnH
      zeta  = zeta  * frnH
      PRINT *, "# dens and zeta have been scaled by:", frnH
   ENDIF

   !--- rop is used in very few chemical reactions, and plays a role only for N+ + H2 -> NH+ + H
   IF (rop < 0.0_dp) THEN                  !--- Compute rop from temperature
      orth = orth_H2(tempr)
      para = 1.0_dp - orth
   ELSE                                    !--- Use value from imput file
      para = 1.0_dp / (1.0_dp + rop)
      orth = rop * para
   ENDIF

   !--- Screen output
   WRITE (iscrn,*) "#"
   WRITE (iscrn,*) "#   F_St_Eq =", F_St_Eq
   WRITE (iscrn,*) "#"
   WRITE (iscrn,*) "#        nH =", densh
   WRITE (iscrn,*) "#       rop =", rop
   WRITE (iscrn,*) "#      tgas =", tempr
   WRITE (iscrn,*) "#       rad =", rad
   WRITE (iscrn,*) "#        Av =", av
   WRITE (iscrn,*) "#       tgr =", tgr
   WRITE (iscrn,*) "#    gr/gas =", ratiog
   WRITE (iscrn,*) "#        rg =", radgr
   WRITE (iscrn,*) "#     rhogr =", rhogr
   WRITE (iscrn,*) "#      zeta =", zeta
   WRITE (iscrn,*) "#"

   !--- Initialize output files
   !--- File names may be built using some internal variables values
   !--- CHARACTER variable ttag gives a convenient conversion

   !WRITE (ttag,'(f6.3)') tempr
   !IF (tempr < 10.0_dp) THEN    ! A leading 0 is convenient for alphabetical sorting
   !   ttag(1:1) = "0"
   !ENDIF
   !WRITE (ttag,'(f4.2)') csco
   !WRITE (ttag,'(f6.4)') frnH
   !WRITE (ttag,'(f6.4)') poc2
   !WRITE (ttag,'(1pe9.3)') poc1
   WRITE (ttag,'(1pe9.3)') rop

   !--- Traditional output file (mostly useless)
   fichi = TRIM(ADJUSTL(outdir)) // TRIM(ADJUSTL(rootnm)) // ".out"
   OPEN  (iwrto, FILE = fichi, STATUS='unknown')

   !--- Full results in columns - used for plots
   fichi = TRIM(ADJUSTL(outdir)) // TRIM(ADJUSTL(rootnm)) // ".graph"
   OPEN  (iwrtg, FILE = fichi, STATUS='unknown')

   !--- Final results for all species (steady states, and max and min time and values
   fichi = TRIM(ADJUSTL(outdir)) // TRIM(ADJUSTL(rootnm)) // ".fin"
   OPEN (iwrtz, FILE = fichi, STATUS="unknown")

   WRITE (iscrn,*) "# O/P, input:", rop, ", fn(T):", orth / para

   !--- Initial grains density (cm-3)
   xngr = ratiog * 3.0_dp * 1.4_dp * densh * xmh &
        / (4.0_dp * xpi * rhogr * radgr**3)

   ! Initialise H2 populations
   abj    = 1.0e-50_dp
   abj(1) = para
   abj(2) = orth
   abj    = abj / SUM(abj)

   !--- Write initial conditions
   WRITE (iwrto,*) " "
   WRITE (iwrto,*) "   State Equation:"
   IF (F_St_Eq == 1) THEN
      WRITE (iwrto,*) "    F_St_Eq =", F_St_Eq, " (T and nH constant)"
   ELSE IF (F_St_Eq == 2) THEN
      WRITE (iwrto,*) "    F_St_Eq =", F_St_Eq, " (isochoric)"
   ELSE IF (F_St_Eq == 3) THEN
      WRITE (iwrto,*) "    F_St_Eq =", F_St_Eq, " (isobaric)"
   ENDIF
   WRITE (iwrto,*) " "

   WRITE (iwrto,2010) densh, tempr, rad, av &
                    , press, tgr, zeta, ratiog, radgr, rhogr, xngr

   WRITE (iwrto,*) " "
   IF (rop < 0.0_dp) THEN
      WRITE (iwrto,*) "   O/P fn(T):", orth / para
   ELSE
      WRITE (iwrto,*) "   O/P input =", rop
   ENDIF
   WRITE (iwrto,*) " "

 2010 FORMAT ("#",8x,'initial values:',/, &
              "#",8x,14("-"),//, &
              5x,'nh     =',1pe12.4,' (cm-3)',/, &
              5x,'tempr  =',0pf12.2,' (K)',//, &
              5x,'rad    =',f12.2,' (* champ moyen)',/, &
              5x,'av     =',f12.2,' (magnitudes)',//, &
              5x,'p/k    =',e12.4,' (k.cm-3)',//, &
              5x,'tgr    =',0pf12.2,' (K)',/, &
              5x,'zeta   =',1pe12.4,' (s-1)',//, &
              5x,'ratiog =',e12.4,'     ',/, &
              5x,'rgrain =',e12.4,' (cm)',/, &
              5x,'rhogr  =',e12.4,' (g.cm-3)',/, &
              5x,'ngrain =',e12.4,' (cm-3)',/)

END SUBROUTINE init

!****************
SUBROUTINE lectur
!****************
!---------------------------------------------------------------------
!    Read chemical species
!    Set initial abundances (coherent with depletions)
!    Read chemical reactions
!
!    Called from main
!---------------------------------------------------------------------

   USE Chem_Aux

   IMPLICIT NONE

   REAL (KIND=dp)     :: xmsp (natom+npart+1)
!--------------------------------------------------------------------------
!
!  Same as in the PDR code
!
!  elements   h       : 01   c       : 02   n       : 03
!  ---------  o       : 04   he      : 05   d       : 06
!             c*      : 07   n*      : 08   o*      : 09
!             ar      : 10   f       : 11   na      : 12
!             mg      : 13   gr      : 14   si      : 15                 ! Used to be al
!             ne      : 16   s       : 17   cl      : 18
!             ca      : 19   fe      : 20   electr  : 21
!             photon  : 22   crp     : 23   phosec  : 24
!             grain   : 25   2h      : 26   xray    : 27
!             elecsec : 28   '      ': 29
!--------------------------------------------------------------------------

   DATA xmsp /  1.007825_dp &       !  1H
             , 12.000000_dp &       ! 12C
             , 14.003074_dp &       ! 14N
             , 15.994915_dp &       ! 16O
             ,  4.002603_dp &       ! 4He
             ,  2.014102_dp &       !  2D
             , 13.003355_dp &       ! 13C
             , 15.000109_dp &       ! 15N
             , 17.999159_dp &       ! 18O
             , 35.967546_dp &       ! 36Ar
             , 18.998403_dp &       ! 19F
             , 22.989770_dp &       ! 23Na
             , 23.985045_dp &       ! 24Mg
             , 26.981541_dp &       ! 27Al
             , 27.976928_dp &       ! 28Si
             , 19.992439_dp &       ! 20Ne
             , 31.972072_dp &       ! 32S
             , 35.4681155_dp &      ! 0.75 x 35Cl + 0.25 x 37Cl
             , 39.962591_dp &       ! 40Ca
             , 55.934939_dp &       ! 56Fe
             ,  0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp &
             ,  0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp/


   CHARACTER (LEN=5)                   :: reff
   CHARACTER (LEN=8)                   :: r1, r2, p1, p2, p3, p4

   REAL (KIND=dp)                      :: cprodu, creact
   REAL (KIND=dp)                      :: dprodu, dreact
   REAL (KIND=dp)                      :: sprodu, sreact
   REAL (KIND=dp)                      :: bilanm, bilanc
   REAL (KIND=dp)                      :: suma
   INTEGER                             :: iprodu, ireact
   INTEGER                             :: i, j, l
   INTEGER                             :: iic
   INTEGER                             :: summ_r, summ_p
   REAL (KIND=dp)                      :: abi_c, abi_o

!--------------------------------------------------------

   !----read species file and set up conservation array

   !--- Count number of species
   fichi = "Data/" // TRIM(ADJUSTL(rootnm)) // "_Speci.dat"
   OPEN  (iread, FILE = fichi, STATUS='old')
   READ  (iread,*)
   nspec = 0
   DO
      READ (iread,*) l
      IF (l == 888) THEN
         EXIT
      ELSE
         nspec = nspec + 1
      ENDIF
   ENDDO
   CLOSE (iread)
   !--- Add electrons
   nspec = nspec + 1
   WRITE (iscrn,*)  " nspec =", nspec

   !--- Allocate variables that depend on the number of species
   !--- This is better than allocating a "large" fixed size as was done previously
   IF (.NOT. ALLOCATED(ael)) THEN
      ALLOCATE (ael(natom,0:nspec+npart))
      ALLOCATE (abrelh2(nspec))
      ALLOCATE (f0(nmhd+nspec))
      ALLOCATE (abin(0:nspec))
      ALLOCATE (charg(0:nspec+npart))
      ALLOCATE (xmol(0:nspec+npart))
      ALLOCATE (enth(0:nspec+npart))
      ALLOCATE (fy(0:nspec+npart))
      ALLOCATE (abev(0:nspec+npart))
      ALLOCATE (observ(0:nspec+npart))
      ALLOCATE (speci(0:nspec+npart))
   ENDIF
   ael     = 0
   abrelh2 = 0.0_dp
   f0      = 0.0_dp
   abin    = 0.0_dp
   enth    = 0.0_dp
   charg   = 0.0_dp
   xmol    = 0.0_dp
   fy      = 0.0_dp
   abev    = 0.0_dp
   observ  = "                       "
   speci   = "        "

   !--- Read species
   fichi = "Data/" // TRIM(ADJUSTL(rootnm)) // "_Speci.dat"
   OPEN  (iread, FILE = fichi, STATUS='old')
   READ  (iread,*)

   !--- Number of neutral, positive ions and negative ions
   nn   = 0
   ni   = 0
   nneg = 0

   i = 0
   DO
      i  = i + 1
!1011 FORMAT (i3,2x,a8,1x,2i2,3i1,1x,4i1,1x,5i1,1x,6i1,1x,i2,1x,1pe10.3,0pf9.3,1x,a23)
      READ (iread,1011) l, speci(i), (ael(j,i), j=1,natom), abin(i), enth(i), observ(i)
      IF (l     == 888) THEN
         EXIT
      ENDIF

      !--- Set minimal abundances (required by the integrator)
      !--- Tolerance criterium seems to include a divide by n(x)
      abin(i) = MAX(1.0e-20_dp, abin(i))

      !--- Identify some useful species (used, e.g., in cooling function or conservation equations)
      !--- Could we find an automatic scheme?
      IF (speci(i) == 'h')     i_h    = i
      IF (speci(i) == 'c')     i_c    = i
      IF (speci(i) == 'n')     i_n    = i
      IF (speci(i) == 'o')     i_o    = i
      IF (speci(i) == 'he')    i_he   = i
      IF (speci(i) == 'd')     i_d    = i
      IF (speci(i) == 'c*')    i_c13  = i
      IF (speci(i) == 'n*')    i_n15  = i
      IF (speci(i) == 'o*')    i_o18  = i
      IF (speci(i) == 'ar')    i_ar   = i
      IF (speci(i) == 'mg')    i_mg   = i
      IF (speci(i) == 'f')     i_f    = i
      IF (speci(i) == 'na')    i_na   = i
      IF (speci(i) == 'gr')    i_gr   = i
      IF (speci(i) == 'si')    i_si   = i
      IF (speci(i) == 'ne')    i_ne   = i
      IF (speci(i) == 's')     i_s    = i
      IF (speci(i) == 'cl')    i_cl   = i
      IF (speci(i) == 'ca')    i_ca   = i
      IF (speci(i) == 'fe')    i_fe   = i
      IF (speci(i) == 'h2')    i_h2   = i
      IF (speci(i) == 'hd')    i_hd   = i
      IF (speci(i) == 'c2')    i_c2   = i
      IF (speci(i) == 'n2')    i_n2   = i
      IF (speci(i) == 'h2o')   i_h2o  = i
      IF (speci(i) == 'oh')    i_oh   = i
      IF (speci(i) == 'ch')    i_ch   = i
      IF (speci(i) == 'ch4')   i_ch4  = i
      IF (speci(i) == 'cd5+')  i_cd5p = i
      IF (speci(i) == 'nh3')   i_nh3  = i
      IF (speci(i) == 'sh')    i_sh   = i
      IF (speci(i) == 'co')    i_co   = i
      IF (speci(i) == 'h+')    i_hp   = i
      IF (speci(i) == 'c+')    i_cp   = i
      IF (speci(i) == 's+')    i_sp   = i
      IF (speci(i) == 'n+')    i_np   = i
      IF (speci(i) == 'nh+')   i_nhp  = i
      IF (speci(i) == 'fe+')   i_fep  = i
      IF (speci(i) == 'h2d+')  i_h2dp = i
      IF (speci(i) == 'c*+')   i_c13p = i
      IF (speci(i) == 'n*+')   i_n15p = i

      !---- Molecular weight
      suma = 0.0_dp
      DO j = 1, natom - 1
         suma = suma + ael(j,i) * xmsp(j)
      ENDDO
      xmol(i) = suma

      IF (ael(natom,i) == 0) THEN                  ! Neutral
         nn = nn + 1
      ELSE IF (ael(natom,i) >= 1) THEN             ! Positive ions
         charg(i) = REAL(ael(natom,i),dp)
         ni = ni + 1
      ELSE IF (ael(natom,i) == -1) THEN            ! negative ions
         charg(i) = -1.0_dp
         nneg = nneg + 1
      ELSE
         PRINT *, "# Error in charge for species", i, speci(i)
         PRINT *, ael(natom,i), " is not permitted"
         STOP
      ENDIF
   ENDDO
   CLOSE (iread)

   !--- SPECIAL CASE - 
   !--- Ajust C+, O and CO abundances to check the impact of initial conditions
   !--- ALternative possibility: ajust on C and not C+
   IF (csco /= -1.0_dp) THEN
      !iic = i_c
      iic = i_cp
      abi_c = abin(iic) + abin(i_co)
      abi_o = abin(i_o) + abin(i_co)
      IF (abi_c > abi_o) THEN
         PRINT *, "# Not enough O to vary initial conditions"
         STOP
      ENDIF
      IF (csco < 0.0_dp .OR. csco > 1.0_dp) THEN
         PRINT *, "# Wrong C/CO ratio"
         STOP
      ENDIF
      abin(iic)  = 1.0e-20_dp + abi_c * csco
      abin(i_co) = 1.0e-20_dp + abi_c * (1.0_dp - csco)
      abin(i_o)  = 1.0e-20_dp + abi_o - abin(i_co)
   ENDIF

   IF (nspec /= nn + ni + nneg + 1) THEN
      PRINT *, " Error in species list!"
      PRINT *, " nspec =", nspec
      PRINT *, "    nn =", nn
      PRINT *, "    ni =", ni
      PRINT *, "  nneg =", nneg
      STOP
   ELSE
      !--- Index of evolution equations (used in diffun)
      nx1 = nmhd + 1
      nx2 = nmhd + nspec
   ENDIF
   observ(nspec) = '                     !'

   !--- Initial abundance of electrons: no grain charge is accounted for
   abin(nspec) = SUM(abin(nn+1:nspec-1) * charg(nn+1:nspec-1))

   !--- Variables used to keep track of maxima and minima of all abundances
   !--- Used to detect phase differences in oscillations
   !--- Used to compute accurate maxima for H at all times
   IF (.NOT. ALLOCATED(abc1)) THEN
      ALLOCATE (abc1(1:nspec),  abc2(1:nspec), abc3(1:nspec))
      ALLOCATE (abmin(1:nspec), abmax(1:nspec))
      ALLOCATE (ttmin(1:nspec), ttmax(1:nspec))
   ENDIF
   tcur  = 0.0_dp
   tecur = 0.0_dp
   abc1  = 0.0_dp
   abc2  = 0.0_dp
   abc3  = 0.0_dp

   !--- Compute depletions, and ajust to required input
   !--- Note: This routine should be changed to something more robust 
   CALL ajust_abin

   !--- Convert initial relative abundances to cm-3
   abin = abin * densh

   WRITE (iwrto,2000)
   WRITE (iwrto,2010) (i, speci(i), xmol(i), abin(i), enth(i), charg(i), i=1,nspec)
   WRITE (iwrto,2020) nn, ni, nneg, nspec

   !----Convert molecular masses in grams
   DO i = 1, nspec
      xmol(i) = xmol(i) * amu
   ENDDO

   !---- Other values are already initialized to 0
   enth(nspec+1) = 290.0_dp            ! Photons
   enth(nspec+2) = 690.0_dp            ! crp
   enth(nspec+3) = 690.0_dp            ! phosec
   charg(nspec)  = -1.0_dp             ! electr

   speci(nspec:nspec+npart) = sp(natom:natom+npart)
   xmol(nspec:nspec+npart)  = xmsp(natom:natom+npart)

   !--- Read chemical reactions
   fichi = "Data/" // TRIM(ADJUSTL(rootnm)) // "_Chemi.dat"
   OPEN  (iread, FILE = fichi, STATUS='old')

   !--- Count number of reactions
   neqt = 0
n_rxn1: DO
      READ (iread,"(A5)",ADVANCE="NO")  reff
      IF (reff == "99999") EXIT n_rxn1         ! End of reaction list
      READ (iread,"(1x,A8)",ADVANCE="NO") r1
         IF (r1 /= "        ") THEN            ! Not a commentary line in chemistry
           neqt = neqt + 1
         ENDIF
         READ (iread,*)                        ! go to begining of next line
   ENDDO n_rxn1
   CLOSE (iread)

   IF (.NOT. ALLOCATED(gamm)) THEN
      ALLOCATE (gamm(neqt))
      ALLOCATE (alpha(neqt))
      ALLOCATE (bet(neqt))
      ALLOCATE (xkrate(neqt))
      ALLOCATE (de(neqt))
      ALLOCATE (summ(neqt))
      ALLOCATE (itype(neqt))
      ALLOCATE (react(neqt,6))
   ENDIF
   gamm   = 0.0_dp
   alpha  = 0.0_dp
   bet    = 0.0_dp
   xkrate = 0.0_dp
   de     = 0.0_dp
   summ   = 0.0_dp
   itype  = 0
   react  = 0

   OPEN  (iread, FILE = fichi, STATUS='old')
   de = 0.0_dp
   i = 1
n_rxn: DO
      READ (iread,"(A5)",ADVANCE="NO")  reff
         IF (reff == "99999") EXIT n_rxn
      READ (iread,"(1x,A8)",ADVANCE="NO") r1
              IF (r1 == "        ") THEN            ! Commentary line in chemistry
                 READ (iread,*)                     ! Go to begining of next line
                 CYCLE                              ! Then read next line
              ENDIF
      ! Read the rest of the reaction
      READ (iread,'(4a8,a7,e8.2,1x,f5.2,1x,f8.1,i4)') &
            r2, p1, p2, p3, p4, gamm(i), alpha(i), bet(i), itype(i)

      react(i,1:6) = 999

      DO j = 1, nspec+npart
         IF (r1 == speci(j)) react(i,1) = j
         IF (r2 == speci(j)) react(i,2) = j
         IF (p1 == speci(j)) react(i,3) = j
         IF (p2 == speci(j)) react(i,4) = j
         IF (p3 == speci(j)) react(i,5) = j
         IF (p4 == speci(j)) react(i,6) = j
      ENDDO
      IF (p2 == sp(natom+npart+1)) react(i,4) = 0
      IF (p3 == sp(natom+npart+1)) react(i,5) = 0
      IF (p4 == sp(natom+npart+1)) react(i,6) = 0

      !--- Test for incomplete reactions
      DO j = 1, 6
         IF (react(i,j) == 999) THEN
            PRINT *, r1, r2, p1, p2, p3, p4
            PRINT *, react(i,j), i, j
            WRITE (iwrto,2080)
            WRITE (iscrn,2080)
            STOP
            CYCLE
         ENDIF
      ENDDO

      !--- Test for conservation of mass
      DO j = 1, natom-1
         summ_r = ael(j,react(i,1)) + ael(j,react(i,2))
         summ_p = ael(j,react(i,3)) + ael(j,react(i,4)) &
                + ael(j,react(i,5)) + ael(j,react(i,6))
         IF (summ_r /= summ_p) THEN
            PRINT *, "# Wrong reaction in chemistry"
            PRINT *, "# For atom:", j, sp(j)
            PRINT *, "# For reaction", i, r1, r2, p1, p2, p3, p4
            PRINT *, "# Reactants:", summ_r
            PRINT *, "# Products :", summ_p
            STOP
         ENDIF
      ENDDO

      !--- Copmpute producs total mass
      !--- and energy balance for each reacion
      dreact = 0.0_dp
      dprodu = 0.0_dp
      summ(i)= 0.0_dp
      bilanm = 0.0_dp
      bilanc = 0.0_dp
      sreact = 0.0_dp
      creact = 0.0_dp
      sprodu = 0.0_dp
      cprodu = 0.0_dp
      DO j = 1, 2
         ireact = react(i,j)
         dreact = dreact + enth(ireact)
         sreact = sreact + xmol(ireact)
         creact = creact + charg(ireact)
      ENDDO
      DO j = 3, 6
         iprodu  = react(i,j)
         dprodu  = dprodu  + enth(iprodu)
         summ(i) = summ(i) + xmol(iprodu)
         sprodu  = sprodu  + xmol(iprodu)
         cprodu  = cprodu  + charg(iprodu)
      ENDDO
      de(i)  = dreact - dprodu
      de(i)  = de(i)  * calev
      bilanm = sreact - sprodu
      bilanc = creact - cprodu
      IF (bilanm >= 1.0e-24_dp .OR. bilanc > 0.0_dp) THEN
         PRINT *, 'bilan  masse reaction', i, sreact, sprodu, bilanm
         PRINT *, 'bilan charge reaction', i, creact, cprodu, bilanc
      ENDIF

      !--- Write chemical reactions
      WRITE (iwrto,2040) reff, r1, r2, p1, p2, p3, p4, gamm(i), alpha(i), bet(i), de(i), i

      IF (ABS(de(i)) > 100.0_dp) THEN
         de(i) = 0.0_dp
      ENDIF
      i = i + 1
   ENDDO n_rxn

   neqt = i - 1
   WRITE (iwrto,2030) neqt
   CLOSE (iread)

   !--- Initial number density and pressure
   xdens = SUM(abin(1:nspec))
   press = xdens * tempr

   WRITE (iwrtg,6000) (speci(i), i = 1, nspec) &
                    , "bthb", "ynn", "bnn", "bee", "bchim", "gamh2" &
                    , "gamgr", "gampeg", "zlambd", "xlamoh", "xlamco" &
                    , "xlah2o", "xlamc", "xlamcp", "xlamo", "xlambn" &
                    , "wthb", "xlambe", "gamrc"

!---------------------------------------------------------------------
 1011 FORMAT (i3,2x,a8,1x,2i2,3i1,1x,4i1,1x,5i1,1x,6i1,1x,i2,1x,1pe10.3,0pf9.3,1x,a23)
 2000 FORMAT ("#",2x,'Species (Molecular weight' &
             ,', iniial abondances, charge)',/, &
              "#",2x,19("-"),/)
 2010 FORMAT (5x,i3,5x,a7,2x,f5.1,2x,1pe10.3,2x,0pf8.3,2x,0pf8.2)
 2020 FORMAT (//,3x,i3,' neutral species', &
               /,3x,i3,' ionised species', &
               /,3x,i3,' negative species', &
               /,3x,i3,' chemical species',//)
 2030 FORMAT (//,3x,i4,' chemical reactions',//)
 2040 FORMAT (2x,a5,1x,a8,a8,3(a8),a7,1pe8.2,2x,0pf6.2,2x,f8.1,2x,f9.3,3x,i4)
 2080 FORMAT ("#",7x,'Wrong reaction !!!')
 6000 FORMAT (1x,'t(Myrs) ', 'zeta    ', 'nh      ', 'T(K)    ', 405a8 )
!---------------------------------------------------------------------

END SUBROUTINE lectur

!********************
SUBROUTINE ajust_abin
!********************
!
! This routine ensure that the total depletions for each atom
!      are exactly equal to requirements.
!------------------------------------------------------------

   USE Chem_Aux

   IMPLICIT NONE

   INTEGER :: j

   !--- Compute initial depletions
   DO j = 1, natom-1
      depl0(j) = SUM(ael(j,1:nspec) * abin(1:nspec))
   ENDDO

   !--- Read required depletions
   fichi = "Data/" // TRIM(ADJUSTL(rootnm)) // "_Deple.dat"
   OPEN  (iread, FILE = fichi, STATUS='old')
   depl = 0.0_dp
   DO j = 1, natom-1
      READ (iread,*) depl(j)
   ENDDO
   CLOSE (iread)

   ! SPECIAL CASE -
   !--- Ajust C and O depletions to explore parameters space
   IF (poc2 /= -1.0_dp) THEN
      depl(4) = poc1 / (1.0_dp + poc2)
      depl(2) = poc2 * poc1 / (1.0_dp + poc2)
   ENDIF

   !--- Set species for conservation equations - Used to find steady state
   iconv(1)  = i_h2
   iconv(2)  = i_c
   iconv(3)  = i_n
   iconv(4)  = i_o
   iconv(5)  = i_he
   iconv(6)  = i_hd
   iconv(7)  = i_c13
   iconv(8)  = i_n15
   iconv(9)  = i_o18
   iconv(10) = i_ar
   iconv(11) = i_f
   iconv(12) = i_na
   iconv(13) = i_mg
   iconv(14) = i_gr
   iconv(15) = i_si
   iconv(16) = i_ne
   iconv(17) = i_s
   iconv(18) = i_cl
   iconv(19) = i_ca
   iconv(20) = i_fep

   IF (.NOT. ALLOCATED(lconv)) THEN
      ALLOCATE (lconv(0:nspec))
   ENDIF
   lconv = 0
   lconv(i_h2)  = 1
   lconv(i_c)   = 1
   lconv(i_n)   = 1
   lconv(i_o)   = 1
   lconv(i_he)  = 1
   lconv(i_hd)  = 1
   lconv(i_c13) = 1
   lconv(i_n15) = 1
   lconv(i_o18) = 1
   lconv(i_ar)  = 1
   lconv(i_f)   = 1
   lconv(i_na)  = 1
   lconv(i_mg)  = 1
   lconv(i_gr)  = 1
   lconv(i_si)  = 1
   lconv(i_ne)  = 1
   lconv(i_s)   = 1
   lconv(i_cl)  = 1
   lconv(i_ca)  = 1
   lconv(i_fep) = 1

   !--- Ajust initial abundances to reach requirements
   !--- H
   IF (depl(1) /= depl0(1)) THEN
      IF (abin(i_h) > - (depl(1) - depl0(1))) THEN
         abin(i_h) = abin(i_h) + (depl(1) - depl0(1))
      ELSE IF (abin(i_h2) > -0.5_dp * (depl(1) - depl0(1))) THEN
         abin(i_h2) = abin(i_h2) + 0.5_dp * (depl(1) - depl0(1))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(1)
         STOP
      ENDIF
   ENDIF

   !--- C
   IF (depl(2) /= depl0(2)) THEN
      IF (abin(i_c) > - (depl(2) - depl0(2))) THEN
         abin(i_c) = abin(i_c) + (depl(2) - depl0(2))
      ELSE IF (abin(i_cp) > - (depl(2) - depl0(2))) THEN
         abin(i_cp) = abin(i_cp) + (depl(2) - depl0(2))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(2)
         STOP
      ENDIF
   ENDIF

   !--- N
   IF (depl(3) < depl0(3)) THEN
      abin(i_n) = abin(i_n) * depl(3) / depl0(3)
      abin(i_n2) = abin(i_n2) * depl(3) / depl0(3)
      abin(i_np) = abin(i_np) * depl(3) / depl0(3)
      depl0(3) = SUM(ael(3,1:nspec) * abin(1:nspec))
   ENDIF
   IF (depl(3) /= depl0(3)) THEN
      IF (abin(i_n) > - (depl(3) - depl0(3))) THEN
         abin(i_n) = abin(i_n) + (depl(3) - depl0(3))
      ELSE IF (abin(i_n2) > - 0.5_dp * (depl(3) - depl0(3))) THEN
         abin(i_n2) = abin(i_n2) + 0.5_dp * (depl(3) - depl0(3))
      ELSE IF (abin(i_np) > - (depl(3) - depl0(3))) THEN
         abin(i_np) = abin(i_np) + (depl(3) - depl0(3))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(3)
         STOP
      ENDIF
   ENDIF

   !--- O
   IF (depl(4) /= depl0(4)) THEN
      IF (abin(i_o) > - (depl(4) - depl0(4))) THEN
         abin(i_o) = abin(i_o) + (depl(4) - depl0(4))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(4)
         STOP
      ENDIF
   ENDIF

   !--- He
   IF (depl(5) /= depl0(5)) THEN
      IF (abin(i_he) > - (depl(5) - depl0(5))) THEN
         abin(i_he) = abin(i_he) + (depl(5) - depl0(5))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(5)
         STOP
      ENDIF
   ENDIF

   !--- D
   IF (depl(6) /= depl0(6)) THEN
      IF (abin(i_d) > - (depl(6) - depl0(6))) THEN
         abin(i_d) = abin(i_d) + (depl(6) - depl0(6))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(6)
         STOP
      ENDIF
   ENDIF

   !--- 13C
   IF (depl(7) /= depl0(7)) THEN
      IF (abin(i_c13) > - (depl(7) - depl0(7))) THEN
         abin(i_c13) = abin(i_c13) + (depl(7) - depl0(7))
      ELSE IF (abin(i_c13p) > - (depl(7) - depl0(7))) THEN
         abin(i_c13p) = abin(i_c13p) + (depl(7) - depl0(7))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(7)
         STOP
      ENDIF
   ENDIF

   !--- 15N
   IF (depl(8) /= depl0(8)) THEN
      IF (abin(i_n15) > - (depl(8) - depl0(8))) THEN
         abin(i_n15) = abin(i_n15) + (depl(8) - depl0(8))
      ELSE IF (abin(i_n15p) > - (depl(8) - depl0(8))) THEN
         abin(i_n15p) = abin(i_n15p) + (depl(8) - depl0(8))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(8)
         STOP
      ENDIF
   ENDIF

   !--- 18O
   IF (depl(9) /= depl0(9)) THEN
      IF (abin(i_o18) > - (depl(9) - depl0(9))) THEN
         abin(i_o18) = abin(i_o18) + (depl(9) - depl0(9))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(9)
         STOP
      ENDIF
   ENDIF

   !--- Ar
   IF (depl(10) /= depl0(10)) THEN
      IF (abin(i_ar) > - (depl(10) - depl0(10))) THEN
         abin(i_ar) = abin(i_ar) + (depl(10) - depl0(10))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(10)
         STOP
      ENDIF
   ENDIF

   !--- F
   IF (depl(11) /= depl0(11)) THEN
      IF (abin(i_f) > - (depl(11) - depl0(11))) THEN
         abin(i_f) = abin(i_f) + (depl(11) - depl0(11))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(10)
         STOP
      ENDIF
   ENDIF

   !--- Na
   IF (depl(12) /= depl0(12)) THEN
      IF (abin(i_na) > - (depl(12) - depl0(12))) THEN
         abin(i_na) = abin(i_na) + (depl(12) - depl0(12))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(12)
         STOP
      ENDIF
   ENDIF

   !--- Mg
   IF (depl(13) /= depl0(13)) THEN
      IF (abin(i_mg) > - (depl(13) - depl0(13))) THEN
         abin(i_mg) = abin(i_mg) + (depl(13) - depl0(13))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(13)
         STOP
      ENDIF
   ENDIF

   !--- Grains
   IF (depl(14) /= depl0(14)) THEN
      print *, depl(14), depl0(14)
      IF (abin(i_gr) > - (depl(14) - depl0(14))) THEN
         abin(i_gr) = abin(i_gr) + (depl(14) - depl0(14))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(14)
         STOP
      ENDIF
   ENDIF

   !--- Si
   IF (depl(15) /= depl0(15)) THEN
      IF (abin(i_si) > - (depl(15) - depl0(15))) THEN
         abin(i_si) = abin(i_si) + (depl(15) - depl0(15))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(15)
         STOP
      ENDIF
   ENDIF

   !--- Ne
   IF (depl(16) /= depl0(16)) THEN
      IF (abin(i_ne) > - (depl(16) - depl0(16))) THEN
         abin(i_ne) = abin(i_ne) + (depl(16) - depl0(16))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(16)
         STOP
      ENDIF
   ENDIF

   !--- S
   IF (depl(17) /= depl0(17)) THEN
      IF (abin(i_s) > - (depl(17) - depl0(17))) THEN
         abin(i_s) = abin(i_s) + (depl(17) - depl0(17))
      ELSE IF (abin(i_sp) > - (depl(17) - depl0(17))) THEN
         abin(i_sp) = abin(i_sp) + (depl(17) - depl0(17))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(17)
         STOP
      ENDIF
   ENDIF

   !--- Cl
   IF (depl(18) /= depl0(18)) THEN
      IF (abin(i_cl) > - (depl(18) - depl0(18))) THEN
         abin(i_cl) = abin(i_cl) + (depl(18) - depl0(18))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(18)
         STOP
      ENDIF
   ENDIF

   !--- Ca
   IF (depl(19) /= depl0(19)) THEN
      IF (abin(i_ca) > - (depl(19) - depl0(19))) THEN
         abin(i_ca) = abin(i_ca) + (depl(19) - depl0(19))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(19)
         STOP
      ENDIF
   ENDIF

   !--- Fe
   IF (depl(20) /= depl0(20)) THEN
      IF (abin(i_fep) > - (depl(20) - depl0(20))) THEN
         abin(i_fep) = abin(i_fep) + (depl(20) - depl0(20))
      ELSE IF (abin(i_fe) > - (depl(20) - depl0(20))) THEN
         abin(i_fe) = abin(i_sp) + (depl(20) - depl0(20))
      ELSE
         PRINT *, " Could not meet depletion requirement for: ", sp(20)
         STOP
      ENDIF
   ENDIF

   !--- Compute new depletions
   !print *, " "
   !DO j = 1, natom-1
   !   depl0(j) = SUM(ael(j,1:nspec) * abin(1:nspec))
   !   print *, j, sp(j), depl0(j), depl(j)
   !ENDDO

END SUBROUTINE ajust_abin

!*******************************
SUBROUTINE diffun (n, z, yy, dy)
!*******************************
!
!   Compute variables derivatives and (optionally) their Jacobian
!--------------------------------------------------------------------

   USE Chem_Aux

   IMPLICIT NONE


   INTEGER, INTENT(IN)                          :: n
   REAL (KIND=dp), INTENT(IN)                   :: z
   REAL (KIND=dp), INTENT(IN), DIMENSION (n)    :: yy
   REAL (KIND=dp), INTENT(OUT), DIMENSION (n)   :: dy

   INTEGER                                      :: i, ii

   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE   :: yn
   REAL (KIND=dp), DIMENSION (:,:), ALLOCATABLE :: jn

   !--------------------------------------------------------------------
   !  fy(i):      i=1 : xdens         i=2 : tempr
   !--------------------------------------------------------------------

   ! Back to normal variables
   fy(1:n) = EXP(yy(1:n))

   xdens = fy(1)
   tempr = fy(2)
   press = xdens * tempr

   !---- Chemical species abundances
   abev(0)       = 0.0_dp
   abev(1:nspec) = fy(nx1:nx2)

   abev(nspec+1) = 1.0_dp
   abev(nspec+2) = 1.0_dp
   abev(nspec+3) = 1.0_dp
   abev(nspec+4) = xngr

   xnh   = abev(i_h)
   xnh2  = abev(i_h2)
   xno   = abev(i_o)
   xnc   = abev(i_c)
   xncp  = abev(i_cp)
   xnh2o = abev(i_h2o)
   xnoh  = abev(i_oh)
   xnco  = abev(i_co)
   xnhe  = abev(i_he)
   xnhpl = abev(i_hp)

   ALLOCATE (yn(1:nspec))
   ALLOCATE (jn(1:nspec,1:nspec))

   IF (F_St_Eq /= 1 .AND. rop < 0.0_dp) THEN                !--- Recompute rop from temperature
      orth = orth_H2(tempr)
      para = 1.0_dp - orth
   ENDIF
   !--- Chemical source terms
   CALL chimie (yn, jn)

   !--- H2 excitation
   CALL coolH2

   !--- Thermal balance source terms
   IF (F_St_Eq /= 1) THEN
      CALL source
   ENDIF

   IF (F_St_Eq == 1) THEN                   ! T & nH consant
      dy(1) = ynn
      dy(2) = 0.0_dp
      DO i = nx1, nx2
         ii = i - nmhd
         dy(i) = yn(ii)
      ENDDO
   ENDIF

   IF (F_St_Eq == 2) THEN                   ! isochoric
      dy(1) = ynn
      dy(2) = bthb - ynn * xk * tempr
      dy(2) = dy(2) / (1.5_dp * xdens * xk)
      DO i = nx1, nx2
         ii = i - nmhd
         dy(i) = yn(ii)
      ENDDO
   ENDIF

   IF (F_St_Eq == 3) THEN                   ! isobaric
      dy(1) = ynn - 0.4_dp * bthb / xk / tempr
      dy(2) = -tempr / xdens * dy(1)
      DO i = nx1, nx2
         ii = i - nmhd
         dy(i) = yn(ii) - 0.4_dp * fy(i) * bthb / (xdens * xk * tempr)
      ENDDO
   ENDIF

   !--- Switch to logarithmic variables
   DO i = 1, n
      dy(i)  = dy(i) / fy(i)
   ENDDO

   DEALLOCATE (yn)
   DEALLOCATE (jn)

END SUBROUTINE diffun

!*************************
SUBROUTINE chimie (yn, jn)
!*************************

!---------------------------------------------------------------------
!   calcul des termes sources des equations d evolution chimique
!   et des termes sources dus aux reactions chimiques intervenant
!   dans les equations mhd.    nn = nbre de neutres
!                              ni = nbre d ions
!                            nneg = nbre d ions negatifs
!                           nspec = nbre total d especes
!                            neqt = nbre de reactions
!                            nmhd = nbre d equations mhd
!
!   Called from diffun
!---------------------------------------------------------------------

   USE Chem_Aux

   IMPLICIT NONE

   REAL (KIND=dp), DIMENSION (:),   INTENT (OUT) :: yn
   REAL (KIND=dp), DIMENSION (:,:), INTENT (OUT) :: jn

   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE    :: up, upde, down

   INTEGER                                       :: ir1, ir2            ! Reactants
   INTEGER                                       :: ip1, ip2, iprodu    ! Products
   INTEGER                                       :: i, j
   REAL (KIND=dp)                                :: cre, fac, rate

   ALLOCATE (up(1:nspec+npart))
   ALLOCATE (upde(1:nspec+npart))
   ALLOCATE (down(1:nspec+npart))
   up   = 0.0_dp
   upde = 0.0_dp
   down = 0.0_dp

   !--- Do not compute Jacobian matrix at each iteration
   IF (F_Jac == 1) THEN
      jn = 0.0_dp
   ENDIF

   !-------------------------------------------------------------------------
   !  "pseudo-species" : nspec + 1 : photon
   !                     nspec + 2 : crp
   !                     nspec + 3 : phosec
   !                     nspec + 4 : grain
   !                     nspec + 5 : 2h
   !                     nspec + 6 : xray
   !                     nspec + 7 : elecsec
   !
   !  ITYPE are similar to the PDR code, but with subtle differences
   !  Please check before making any modification.
   !-------------------------------------------------------------------------

   DO i = 1, neqt
      ir1 = react(i,1)
      ir2 = react(i,2)
      ip1 = react(i,3)
      ip2 = react(i,4)

      rate = 0.0_dp
      fac = bet(i) / tempr

      IF (F_St_Eq == 1 .AND. ts /= 0.0_dp) THEN
         rate = xkrate(i)
         IF (itype(i) == 0) THEN
            IF (((ir1 == i_h) .OR. (ir2 == i_h))) THEN
               rate = rate / abev(i_h)
            ELSE IF ((ir1 == i_d) .AND. (ir2 == i_d)) THEN
               rate = rate / abev(i_d)
            ENDIF
         ENDIF
      ELSE
         ! Formation of H2, HD and D2 "Old Style"
         IF (itype(i) == 0) THEN
            IF (((ir1 == i_h) .AND. (ir2 == i_h))) THEN
               xkrate(i) = 0.5_dp * gamm(i) * densh * (tempr / 300.0_dp)**alpha(i)
               rate      = xkrate(i) / abev(i_h)
            ELSE IF ((ir1 == i_h) .AND. (ir2 == i_d) .OR. ((ir1 == i_d) .AND. (ir2 == i_h))) THEN
               xkrate(i) =          gamm(i) * densh * (tempr / 300.0_dp)**alpha(i) / SQRT(2.0_dp)
               rate      =          xkrate(i) / abev(i_h)
            ELSE IF ((ir1 == i_d) .AND. (ir2 == i_d)) THEN
               xkrate(i) = 0.5_dp * gamm(i) * densh * (tempr / 300.0_dp)**alpha(i) / SQRT(2.0_dp)
               rate      = xkrate(i) / abev(i_d)
            ENDIF

         ! Destruction by cosmic rays (crp)
         ELSE IF (itype(i) == 1) THEN
            rate = gamm(i) * zeta * ((tempr / 300.0_dp)**alpha(i))

         ! Destruction by secondary photons (phosec)
         ELSE IF (itype(i) == 2) THEN
            rate = gamm(i) * zeta * ((tempr / 300.0_dp)**alpha(i))

         ! Radiative association
         ELSE IF (itype(i) == 3) THEN
            rate = gamm(i) * EXP(-fac) * ((tempr / 300.0_dp)**alpha(i))

         ! Ordinary binary reaction
         ELSE IF (itype(i) == 4) THEN
            IF (fac < 0.0_dp) THEN
              ! reactions pepe
              !rate = gamm(i) * ((tempr / 300.0_dp)**alpha(i))
              !rate = rate * EXP(1./fac)
               rate = 0.0_dp
            ELSE
               rate = gamm(i) * EXP(-fac) * ((tempr / 300.0_dp)**alpha(i))
            ENDIF

         ! Photodestruction
         ELSE IF (itype(i) == 5) THEN
            rate = gamm(i) * EXP(-bet(i) * av) * rad

         ! Specific rates
         ELSE IF (itype(i) == 101) THEN    ! N+ + H2 - Dislaire et al. (2012)
            IF ((ir1 == i_np .AND. ir2 == i_h2) .OR. (ir1 == i_h2 .AND. ir2 == i_np)) THEN
               rate = para * 8.35e-10_dp * EXP(-168.5_dp / tempr) &
                    + orth * 4.20e-10_dp * (tempr / 300.0_dp)**(-0.17_dp) * EXP(-44.5_dp / tempr)
               !rate = 1.700e-15_dp
            ELSE IF ((ir1 == i_n15p .AND. ir2 == i_h2) .OR. (ir1 == i_h2 .AND. ir2 == i_n15p)) THEN
               rate = gamm(i) * (para * EXP(-bet(i) / tempr) &
                    + orth * 0.5_dp * EXP(-39.7_dp / tempr) * (tempr / 300.0_dp)**alpha(i))
            ENDIF
         ELSE IF (itype(i) == 102) THEN    ! H2D+ + H2 - Hugo et al. (2009)
            !rate = 2.10e-09_dp * para * EXP(-232.0_dp / tempr) &
            !     + 2.10e-09_dp * orth * EXP(-61.5_dp / tempr)
            rate = 2.46e-10_dp * para * EXP(-226.5_dp / tempr) &
                 + 1.48e-10_dp * orth * EXP(-58.8_dp / tempr)

         ! Adsorption on grains and neutralization
         ELSE IF (itype(i) == 13) THEN
            rate = gamm(i) * xpi * radgr * radgr * SQRT(8.0_dp * xk * tempr / xmol(ir1) / xpi)

         ! Photodesorption from grains
         ELSE IF (itype(i) == 17) THEN
            rate = gamm(i) * rad * EXP(-2.0_dp * av) * xpi * radgr * radgr &
                 + gamm(i) *  EXP(-200.0_dp / tgr) &
                 + gamm(i) * zeta / 1.0e-17_dp * 1000.0_dp * 1.0e-2_dp *xpi * radgr * radgr
         ENDIF
         IF (itype(i) /= 0) THEN
            xkrate(i) = rate
         ENDIF
      ENDIF

      ! Source terms due to chemistry
      cre = rate * abev(ir1) * abev(ir2)
      !--- Destruction terms
      down(ir1) = down(ir1) + cre
      down(ir2) = down(ir2) + cre
      !--- Creation terms
      DO j = 3, 6
         iprodu = react(i,j)
         IF (iprodu == 0) CYCLE
         up(iprodu)   = up(iprodu)   + cre
         upde(iprodu) = upde(iprodu) + cre * de(i) * (1.0_dp - xmol(iprodu) / summ(i))
      ENDDO

      !--- Jacobian matrix
      IF (F_Jac == 1) THEN
         jn(ir1,ir1) = jn(ir1,ir1) - rate * abev(ir2)
         IF (ir2 <= nspec) THEN
            jn(ir2,ir1) = jn(ir2,ir1) - rate * abev(ir2)
            jn(ir1,ir2) = jn(ir1,ir2) - rate * abev(ir1)
            jn(ir2,ir2) = jn(ir2,ir2) - rate * abev(ir1)
         ENDIF
         DO j = 3, 6
            iprodu = react(i,j)
            IF (iprodu == 0 .OR. iprodu > nspec) CYCLE
            jn(iprodu,ir1) = jn(iprodu,ir1) + rate * abev(ir2)
            IF (ir2 <= nspec) THEN
               jn(iprodu,ir2) = jn(iprodu,ir2) + rate * abev(ir1)
            ENDIF
         ENDDO
      ENDIF
   ENDDO

   !--- sommation des taux : (creation) - (destruction)
   !    termes sources pour chaque espece chimique : (nombre)/cm3/s
   yn(1:nspec) = up(1:nspec) - down(1:nspec)

   !--- sommation sur les especes chimiques : termes sources des eq. mhd
   !                        (nombre, energie)
   ynn   = SUM(yn(1:nspec))
   bchim = SUM(upde(1:nspec)) * everg

   DEALLOCATE (up)
   DEALLOCATE (upde)
   DEALLOCATE (down)

END SUBROUTINE chimie

!****************
SUBROUTINE source
!****************

   USE Chem_Aux

   IMPLICIT NONE

!---------------------------------------------------------------------
!   appele par diffun (z,y,dy)
!   calcul des termes sources : bthb
!---------------------------------------------------------------------

   REAL (KIND=dp) :: gamphi, rate, rc, ups
   REAL (KIND=dp) :: xlc1, xlc2

!---------------------------------------------------------------------
!   preparation des fonctions de refroidissement :
!   ********************************************

!                    *******************
!--------------------* refroidissement *--------------------------------
!                    *******************
!---neutre-neutre
!   -------------

!---refroidissement par excitation de o par h et h2
   xlamo = 2.4e1_dp * EXP(-228.0_dp / tempr) + 7.0_dp * EXP(-326.0_dp / tempr)
   xlamo = xlamo * 1.0e-26_dp * tempr**0.5_dp * xno * (xnh + xnh2 / SQRT(2.0_dp))

!---refroidissement par excitation de c+ par h et h2
   xlamcp = 22.0e-24_dp * EXP(-92.0_dp / tempr) * xncp * xnh
   xlamcp = xlamcp + 15.0e-24_dp * EXP(-92.0_dp / tempr) * xncp * xnh2

!---refroidissement par excitation de c par h et h2
!           limite a basse densite (n < 1000 cm-3)
   xlc1  = (1.4_dp + 1.8e-2_dp * SQRT(tempr)) * EXP(-23.0_dp / tempr) * (xnh + xnh2 / 10.0_dp) &
         + (3.8_dp + 0.11_dp   * SQRT(tempr)) * EXP(-62.0_dp / tempr) * (xnh + xnh2)
   xlc1  = 1.0e-24_dp * xlc1 * xnc
!           limite a haute densite (n > 1000 cm-3)
   xlc2  = 0.76_dp * EXP(-23.0_dp / tempr) + 7.1_dp * EXP(-62.0_dp / tempr)
   xlc2  = xlc2 / (1.0_dp + 3.0_dp * EXP(-23.0_dp / tempr) + 5.0_dp * EXP(-62.0_dp / tempr))
   xlc2  = 1.0e-21_dp * xlc2 * xnc
   xlamc = MIN(xlc1,xlc2)

!---refroidissement par excitation de h2o, oh, c13o et par h et h2
   xlah2o = 1.0e-26_dp * tempr * EXP(-35.0_dp / tempr)  * xnh2o * (xnh + xnh2 / SQRT(2.0_dp))
   xlamoh = 1.0e-26_dp * tempr * EXP(-120.0_dp / tempr) * xnoh * (10.0_dp * xnh + xnh2 / SQRT(2.0_dp))

!---contribution de c13o :c13 / c12 = 1/90
   xlamco = 0.1_dp * 1.0e-26_dp * tempr * EXP(-10.0_dp / tempr) * xnco * (xnh + xnh2 / SQRT(2.0_dp))

   xlambn = xlamo + xlamcp + xlamc + xlah2o + xlamco + xlamoh

   !---perte d energie due au refroidissement : zlambd
   !--- "w" has been computed in coolH2, before a call to source
   zlambd = -xlambn - wthb

   ! 12 VI 2020 - JLB - Try scaling cooling to keep a lower temperature
   zlambd = 1.0_dp * zlambd

!---electron-neutre
!   ---------------

!---excitation 2p1/2 - 2p3/2 de c+ par collision electronique
   xlambe = 5.50e-20_dp * EXP(-92.0_dp / tempr) / tempr**0.5_dp
   ups    = 1.8_dp + 1.1e-4_dp * tempr
   xlambe = xlambe * MIN(2.9_dp,ups)
   xlambe = xlambe * xncp * abev(nspec)

!                       *************
!-----------------------* chauffage *-----------------------------------
!                       *************
!---neutres
!   -------

!---chauffage par effet photoelectrique sur les grains (black,1987):
   gampeg = 4.0e-26_dp * (xnh + 2.0_dp * xnh2) * rad * EXP(-2.5_dp * av)

!---chauffage/refroidissement par equ. thermique avec les grains

! (1) black (1987, interstellar processes, p.731, reidel)
!  gamgr = 2.1e-33_dp * SQRT(tempr) * (tgr - tempr) * (xnh + 2.0_dp * xnh2)**2

! (2) tielens et hollenbach (1987, ap.j. 291, 722)
!  gamgr = 3.5e-34_dp * SQRT(tempr) * (tgr - tempr) * (xnh + 2.0_dp * xnh2)**2
!  gamgr = gamgr * 0.2_dp

! (3) en tenant compte des caracteristiques des grains (voir notes)
   gamgr = 7.02e-36_dp * ratiog / rhogr / radgr * SQRT(tempr) * (tgr - tempr)
   gamgr = gamgr * 0.28_dp
   gamgr = gamgr * (xnh + xnh2) * (xnh + 2.0_dp * xnh2)

!---chauffage par formation de h2 sur les grains
! old : gamh2 = 4.7d-18 * SQRT(tempr) * xnh*(xnh+2.*xnh2) * 1.5 * everg

   rate  = 2.2e-19_dp * ratiog / rhogr / radgr * SQRT(tempr / 300.0_dp)
   gamh2 = rate * xnh * (xnh + 2.0_dp * xnh2) * 1.5_dp * everg

!---bilan d energie des neutres : bnn
   bnn = zlambd + gampeg + gamgr + gamh2

!---electrons
!   ---------

!---chauffage du a l ionisation par les rayons cosmiques
!                (spitzer and scott, 1969)
   rc    = 32.0_dp - 7.0_dp * LOG10(xdens / abev(nspec))
   rc    = MAX(5.7_dp,rc)
   gamrc = zeta * rc * everg * xdens
   !print *, " ---", xdens, densh * SUM(depl)
! pour donnees umist
!  gamrc = zeta * rc * everg * xdens * 1.0e-17_dp

!---chauffage du a la photoionisation (1 ev par photoreaction)
!                   (black, 1987)
!  gamphi = 2.0e-22_dp * xnc
   gamphi = 0.0_dp

!---bilan d energie des electrons : bee
   bee = gamrc + gamphi - xlambe

!                 **************************
!-----------------* bilan thermique du gaz *---------------------------
!                 **************************

   bthb = bnn + bee + bchim

END SUBROUTINE source

!****************
SUBROUTINE coolH2
!****************

!---------------------------------------------------------------------
!   xnph2 is no. density of para-h2 molecules (cm-3)
!   wthb is computed cooling rate (erg cm-3 s-1)

!   include 10 levels of para-h2,  j=0 (2), 18
!       and 10 levels of ortho-h2, j=1 (2), 19

!   called from sub. source
!---------------------------------------------------------------------

   USE Chem_Aux

   IMPLICIT NONE

   !---einstein A values and rotational constant br for h2
   REAL (KIND=dp), DIMENSION (n_rot-1) :: aein
   REAL (KIND=dp)                      :: br
   DATA aein / 0.0_dp,     2.94e-11_dp, 4.76e-10_dp, 2.76e-9_dp, 9.84e-9_dp &
             , 2.64e-8_dp, 5.88e-8_dp,  1.14e-7_dp,  2.00e-7_dp, 3.24e-7_dp &
             , 4.90e-7_dp, 7.03e-7_dp,  9.64e-7_dp,  1.27e-6_dp, 1.62e-6_dp &
             , 2.00e-6_dp, 2.41e-6_dp,  2.83e-6_dp,  3.26e-6_dp /
   DATA br / 170.5_dp /

   !---coefficients for h2(j)-h2(1) rates, from danby & flower(1986)
   REAL (KIND=dp), DIMENSION (n_rot-1) :: ah2, bh2, ch2
   DATA ah2 / 0.0_dp,      3.166e-4_dp, 2.377e-5_dp, 1.940e-6_dp, 1.674e-7_dp &
            , 1.603e-8_dp, 2.322e-9_dp, 12*0.0_dp /
   DATA bh2 / 0.0_dp,      1.1652_dp,   1.512_dp,    1.831_dp, 2.113_dp, 2.3757_dp &
            , 2.575_dp,    12*0.0_dp /
   DATA ch2 / 0.0_dp,      3.068e-2_dp, 1.124e-2_dp, 5.425e-3_dp, 1.981e-3_dp &
            , 7.527e-4_dp, 2.766e-4_dp, 12*0.0_dp /

   !---cw gives ratios of para/ortho rates (flower & danby)
   REAL (KIND=dp), DIMENSION (n_rot-1) :: cw
!  DATA cw / 0.0_dp, 0.81_dp, 0.48_dp, 0.89_dp, 0.89_dp, 0.89_dp, 0.89_dp, 12*0.0_dp /

   !---fit to he-h2 rate coefficients of schaefer
   REAL (KIND=dp), DIMENSION (n_rot-1) :: aa, bb
   DATA aa / 0.0_dp, 1.210_dp, 1.568_dp, 1.852_dp, 2.124_dp, 2.255_dp, 13*0.0_dp /
   DATA bb / 0.0_dp, -14.458_dp, -15.564_dp, -16.619_dp, -17.670_dp, -18.331_dp, 13*-20.0_dp/

   REAL (KIND=dp) :: xnph2

   REAL (KIND=dp)         :: alpha(n_rot-1)
   INTEGER, DIMENSION (2) :: indx

   REAL (KIND=dp) :: b, c, e, e01, fac, p, sigvh, sigvh2, sigvhe, suma
   REAL (KIND=dp) :: tem1, vbar, wvib, x, xkt, theta
   INTEGER        :: i, k, j, jp, l

   DATA indx / 0, 1 /

!---rotational constant b
   b = xk * br * 0.5_dp

!---rapport ortho/para lu 
   xnph2 = xnh2 * para

!---compute kt
   xkt = xk * tempr

   !---mean thermal speed
   !   with reduced mass (g) for collisions with h atoms
   vbar = SQRT(2.54648_dp * xkt / (2.0_dp * amu / 3.0_dp))

   wthb = 0.0_dp
   DO k = 1, 2
      DO i = 2, 10
         j  = 2 * (i-1) + indx(k)
         jp = j - 2
         alpha(j) = 0.0_dp

!---eq.1 of e+wthb
         e = xkt + j * (j+1) * b
         c = 2.0e-16_dp * (2 * jp + 1) * SQRT((e / b - jp * (jp + 1)) / (e / b - j * (j + 1)))
         theta = (5.01_dp / e + 0.1187_dp / b) * ABS(j * (j + 1) - jp * (jp + 1)) * b
         c = c * EXP(-theta)

!---eq.4 of e+wthb
         c   = 1.3_dp * c * vbar
         fac = (2 * j - 1) * 2 * b / xkt
         IF (fac <= 1.0e2_dp) THEN
!           d = (2 * j + 1) * c * EXP(-fac) / (2 * j - 3)

!---multiply by atomic hydrogen density
            c = c * xnh

!---add rate of excitation by he and h2
            IF (aa(j) /= 0.0_dp) THEN
               c = c + 10.0_dp**(aa(j) * LOG10(tempr) + bb(j)) * xnhe

            ENDIF
!---work out h2 - h2 rates
            tem1 = ah2(j) * tempr**bh2(j) + ch2(j)
            tem1 = tem1 * 1.0e-11_dp
!-----------------------------------------------------
!     c = c + tem1 * (xnh2 + (cw(j) - 1.0_dp) * xnph2)
! line above weighted o/p h2
!-----------------------------------------------------
             c = c + tem1 * xnh2

!---eq.9 of e+wthb (use 170.5 in exponent, and not 175)
            alpha(j) = (2 * j + 1) / (1.0_dp + aein(j) / c) * EXP(-fac) / (2 * j - 3)
         ENDIF
      ENDDO

!---population densities in ground states j=0 and j=1
      suma = 1.0_dp
      DO i=1,9
         j = 10 - i
         j = 2 * j + indx(k)
         suma = 1.0_dp + alpha(j) * suma
      ENDDO
      x = ABS(indx(k) * xnh2 - xnph2) / suma
      l = indx(k) + 1
      abj(l) = x

!---population densities in excited states j=2 (2), 18
!                                      and j=3 (2), 19
!---and cooling rates : wthb (erg cm-3 s-1)
      DO i = 2, 10
         j = 2 * (i - 1) + indx(k)
         IF (x > 1.0e-60_dp) THEN
            x = alpha(j) * x
            wthb = wthb + x * aein(j) * (2 * j - 1)
            abj(j+1) = x
         ENDIF
      ENDDO
   ENDDO
   wthb = wthb * 2.0_dp * b

!---refroidissement vibrationnel:  wvib  (erg.cm-3.s-1)
!   ****************************

!   (attention, pour les modeles denses, il faudra tenir compte
!       de la desexcitation collisionnelle, negligee ici.)

!---taux de desexcitation   (cm3.s-1)
   sigvh2 = tempr * 10.0_dp**(-34.74_dp / tempr**(1.0_dp/3.0_dp) - 13.18_dp)
   sigvhe = tempr * 10.0_dp**(-41.35_dp / tempr**(1.0_dp/3.0_dp) - 12.88_dp)
   sigvh  = SQRT(tempr) * 10.0_dp**(-434.3_dp / tempr - 12.0_dp)
!---probabilite de desexcitation   (s-1)
   p = xnh2 * sigvh2 + xnhe * sigvhe + xnh * sigvh
!---probabilite d excitation   (s-1)
   p = p * EXP(-5986.0_dp / tempr)

   e01  = 5986.0_dp * xk
   wvib = p * xnh2 * e01
   wthb = wthb + wvib

END SUBROUTINE coolH2

!*******************
SUBROUTINE steady_st
!*******************
!
! Compute steady state using a Newton-Raphson scheme
! Close to the algorithm used in PDR code
!---------------------------------------------------

   USE Chem_Aux
   USE qeispack

   IMPLICIT NONE

   REAL (KIND=dp), DIMENSION (:),   ALLOCATABLE :: abst, abold
   REAL (KIND=dp), DIMENSION (:),   ALLOCATABLE :: abt
   REAL (KIND=dp), DIMENSION (:),   ALLOCATABLE :: dy
   REAL (KIND=dp), DIMENSION (:,:), ALLOCATABLE :: jn, jacf
   REAL (KIND=dp)                               :: errx, errf
   REAL (KIND=dp)                               :: seux, seuf
   INTEGER                                      :: i, j, k

   INTEGER                                      :: nrhs, INFO
   INTEGER, DIMENSION (nspec)                   :: indx
   REAL (KIND=dp), DIMENSION (nspec,1)          :: bb
   REAL (KIND=dp)                               :: dum, duma
   REAL (KIND=dp)                               :: damp = 0.7_dp

   !--- Variables used to call LAPACK routine DGEEVX
   CHARACTER (LEN=1)                              :: BALANC = 'B'
   CHARACTER (LEN=1)                              :: JOBVL = 'V'
   CHARACTER (LEN=1)                              :: JOBVR = 'V'
   CHARACTER (LEN=1)                              :: SENSE = 'B'
   REAL (KIND=dp), DIMENSION(:),   ALLOCATABLE    :: WR
   REAL (KIND=dp), DIMENSION(:),   ALLOCATABLE    :: WI
   REAL (KIND=dp), DIMENSION(:,:), ALLOCATABLE    :: VL, VR
   REAL (KIND=dp), DIMENSION(:),   ALLOCATABLE    :: WORK
   REAL (KIND=dp), DIMENSION(:),   ALLOCATABLE    :: ASCALE
   REAL (KIND=dp)                                 :: ABNRM
   REAL (KIND=dp), DIMENSION(:),   ALLOCATABLE    :: RCONDE, RCONDV
   INTEGER, DIMENSION(:),   ALLOCATABLE           :: IWORK
   INTEGER                                        :: LWORK
   INTEGER                                        :: ILO, IHI
   INTEGER                                        :: INFOJ

   !--- Variables used for EISPACK routine rg
   REAL (KIND=qp), DIMENSION(:,:), ALLOCATABLE    :: aaj, zzz
   REAL (KIND=qp), DIMENSION(:),   ALLOCATABLE    :: wwr, wwi
   INTEGER       , DIMENSION(:),   ALLOCATABLE    :: rnk1, rnk2
   INTEGER                                        :: matz, ierr

   ALLOCATE (abst(1:nspec))
   ALLOCATE (abold(1:nspec))
   ALLOCATE (abt(1:nspec))
   ALLOCATE (dy(1:nspec))
   ALLOCATE (jn(1:nspec,1:nspec))
   ALLOCATE (jacf(1:nspec,1:nspec))

   ALLOCATE (wwr(nspec), wwi(nspec))
   ALLOCATE (aaj(nspec,nspec), zzz(nspec,nspec))
   ALLOCATE (rnk1(nspec), rnk2(nspec))

   LWORK = 2 * nspec * nspec + 3
   ALLOCATE (WR(nspec), WI(nspec))
   ALLOCATE (VL(nspec,nspec), VR(nspec,nspec))
   ALLOCATE (WORK(LWORK))
   ALLOCATE (ASCALE(nspec))
   ALLOCATE (RCONDE(nspec), RCONDV(nspec))
   ALLOCATE (IWORK(2*nspec-2))

   !--- Set accuracy thresholds (NB: no test has been performed yet)
   seux = 1.0e-8_dp
   seuf = 1.0e-8_dp
   F_Jac = 1
   nrhs = 1

   !--- Compute maximum possible abundances for all species
   abt(:) = 0.0_dp
   DO i = 1, nspec-1
      duma = densh
      DO j = 1, natom-1
         IF (ael(j,i) == 0) THEN
            CYCLE
         ELSE
            duma = MIN(duma,densh*depl(j)/ael(j,i))
         ENDIF
      ENDDO
      abt(i) = duma
   ENDDO
   ! Use a wide margin for electr abundance
   abt(nspec) = 2.0_dp * SUM(abev(nn+1:nspec-1) * charg(nn+1:nspec-1))

   !--- Save abundances for a possible restart of evolution
   abst(1:nspec) = abev(1:nspec)

   DO i = 1, 100
      abold = abev(1:nspec)
      !--- Compute chemical balance and Jacobian
      CALL chimie (dy, jn)
      jacf = jn
      aaj(1:nspec,1:nspec) = REAL(jacf(1:nspec,1:nspec),qp)

      !--- Replace by conservation equations for selected species
      DO j = 1, natom-1
         IF (depl(j) == 0.0_dp) CYCLE
         dy(iconv(j)) = SUM(abev(1:nspec)*ael(j,1:nspec)) - depl(j) * densh
         DO k = 1, nspec
            jn(iconv(j),k) = ael(j,k)
         ENDDO
      ENDDO
      !--- Special case of electrons
      dy(nspec) = abev(nspec) - SUM(abev(nn+1:nspec-1) * charg(nn+1:nspec-1))
      DO k = 1, nspec-1
         jn(nspec,k) = -ael(natom,k)
      ENDDO
      jn(nspec,nspec) = 1.0_dp

      !--- Solve using LAPACK
      bb(:,1) = dy
      CALL DGESV (nspec, nrhs, jn, nspec, indx, bb, nspec, INFO)

      !--- Limit variations to gas density for ALL variables
      dum = SUM(ABS(bb(1:nspec,1)))
      IF (dum > densh) THEN
         bb(1:nspec,1) = bb(1:nspec,1) * densh / dum
      ENDIF

      !--- Upgrade abundances
      !--- Note: this is heavily based on the PDR code
      errx = 0.0_dp
      DO k = 1, nspec
         IF (abev(k) > MAX(istiny,-bb(k,1)) .OR. bb(k,1) > 0.0_dp) THEN
            IF (abev(k) > 1.0e45_dp*istiny) THEN
               errx = errx + ABS(bb(k,1)) / (abold(k) + istiny)
            ENDIF
            IF (lconv(k) == 0) THEN
               abev(k) = abev(k) - damp * bb(k,1)
            ELSE
               abev(k) = abev(k) - bb(k,1)
            ENDIF
            IF (abev(k) > abt(k)) THEN
               abev(k) = (abt(k) + abold(k)) * 0.5_dp
            ELSE IF (abev(k) < 0.0_dp) THEN
               abev(k) = abold(k) / 10.0_dp
            ENDIF
         ENDIF
         abev(k) = MAX(abev(k),istiny)
      ENDDO

      IF (MOD(i,5) == 1) THEN
         abev(1:nspec) = (abev(1:nspec) + abold(1:nspec)) * 0.5_dp
      ENDIF

      errf = SUM(ABS(dy))
      !PRINT *, i, " x:", errx, " f:", errf, SUM(ABS(dy(1:nspec)/abev(1:nspec))), abev(1), dy(1)
      IF (errx < seux .AND. errf < seuf) EXIT
   ENDDO
   PRINT *, " "
   PRINT *, " Steady state has been computed:"
   !PRINT *, i, " errx:", errx, " errf:", errf, SUM(ABS(dy(1:nspec)/abev(1:nspec))), abev(1), dy(1)
   PRINT *, i, " errx:", errx, " errf:", errf, SUM(ABS(dy(1:nspec)/abev(1:nspec)))

   !--- Output to a specific file, and within the "graph" file for comparison
   WRITE(iwrtz,'("N Species Ab-SS t-min Ab-min t-max Ab-max I-ep Temper Rap Del-t")')
   DO i = 1, nspec
      WRITE (iwrtz,*) i, speci(i), abev(i), ttmin(i), abmin(i), ttmax(i), abmax(i) &
                 , densh/zeta, tempr, abmax(i) / abmin(i), ttmax(i) - ttmin(i)
   ENDDO
   WRITE (iwrtz,'(/)')
   WRITE (iwrtz,*) ttemin, temin, ttemax, temax, densh, zeta, temax / temin, ttemax - ttemin
   CLOSE (iwrtz)
   WRITE (iwrtg,'(" ")')
   WRITE (iwrtg,'(405(1pe16.8))') ts, zeta, densh, tempr, (abev(i), i=1,nspec) &
                       , bthb, ynn, bnn, bee, bchim
   WRITE (iwrtg,'(" ")')

   !--- Compute eigenvalues
   !--- Use 2 different algorithms to check for accuracy
   IF (F_EIGEN == 1) THEN

      !--- Call LAPACK
      CALL DGEEVX(BALANC, JOBVL, JOBVR, SENSE, nspec, jacf, nspec, WR, WI, VL, nspec, VR, &
                  nspec, ILO, IHI, ASCALE, ABNRM, RCONDE, RCONDV, WORK, LWORK, IWORK, INFOJ)
      !PRINT *, INFOJ, ABNRM, MAXVAL(RCONDE), MAXVAL(RCONDV), WORK(1), LWORK
      CALL sortwri(WR,rnk1)
      PRINT *, " "

      !--- Call eispack
      matz = 0
      CALL rg ( nspec, aaj, wwr, wwi, matz, zzz, ierr )
      !print *, ierr
      CALL sortwri(REAL(wwr,dp),rnk2)
      print *, " "

      PRINT *, " Eigenvalues have been computed"
      PRINT *, "    Printing unstable values (positive real part) and oscillating values (non 0 imaginary part)"
      DO i = 1, nspec
         !--- Eigenvalues too close to 0 are not reliable => set them to exactly 0
         IF ( ABS(ABS(WR(rnk1(i))) - ABS(wwr(rnk2(i)))) / (ABS(WR(rnk1(i))) + ABS(wwr(rnk2(i)))) > 1.0e-3_dp) THEN
            WR(rnk1(i))  = 0.0_dp
            WI(rnk1(i))  = 0.0_dp
            wwr(rnk2(i)) = 0.0_dp
            wwi(rnk2(i)) = 0.0_dp
         ENDIF
         !--- Print positive eigenvalues => unstable steady state
         IF (wwr(rnk2(i)) > 0.0_dp) THEN
            WRITE (iscrn,'(2i4,1p,2e17.8,i4,2e17.8)') &
                  i, rnk1(i), WR(rnk1(i)), WI(rnk1(i)), rnk2(i), wwr(rnk2(i)), wwi(rnk2(i))
         ENDIF
         !--- Print eigenvalues with non 0 imaginary part (only once: positive root)
         IF (wwi(rnk2(i)) > 0.0_dp) THEN
            WRITE (iscrn,'(2i4,1p,2e17.8,i4,3e17.8)') &
                  i, rnk1(i), WR(rnk1(i)), WI(rnk1(i)), rnk2(i), wwr(rnk2(i)), wwi(rnk2(i)) &
                , 1.0_dp / (wwi(rnk2(i)) * Myr)
         ENDIF
         !IF (WR(rnk1(i)) > 0.0_dp) THEN
         !   PRINT *, " "
         !   DO j = 1, nspec
         !      WRITE (iscrn,*) j, speci(j), VR(j,rnk1(i)) / abev(j)
         !   ENDDO
         !ENDIF
      ENDDO
      print *, " "
   ENDIF

   !--- Restore evolution state
   abev(1:nspec) = abst(1:nspec)
   F_Jac = 0

   DEALLOCATE (abst)
   DEALLOCATE (abold)
   DEALLOCATE (abt)
   DEALLOCATE (dy)
   DEALLOCATE (jn)
   DEALLOCATE (jacf)

   DEALLOCATE (wwr, wwi)
   DEALLOCATE (aaj, zzz)
   DEALLOCATE (rnk1, rnk2)

END SUBROUTINE steady_st

!*************************
SUBROUTINE max_ht (tt, pp)
!*************************
!
! Use a second order polynomial interpolation to find a better approximation of max
! Only H is used here. The wave form is simple, which leads to a better precision
! This can be changed in main.
!-----------------------------

   USE Chem_Aux

   IMPLICIT NONE

   REAL (KIND=dp), INTENT (OUT)               :: tt, pp
   REAL (KIND=dp)                             :: t1, t2, t3
   REAL (KIND=dp)                             :: y1, y2, y3
   REAL (KIND=dp)                             :: l1, l2, l3

   t1 = tcur(1)
   t2 = tcur(2)
   t3 = tcur(3)
   y1 = abc1(i_h)
   y2 = abc2(i_h)
   y3 = abc3(i_h)

   !--- Time of maximum
   tt = 0.5_dp * (y1*(t2**2-t3**2) - y2*(t1**2-t3**2) + y3*(t1**2-t2**2)) &
      / (y1*(t2-t3) - y2*(t1-t3) + y3*(t1-t2))

   l1 = (tt-t2) * (tt-t3) / ((t1-t2) * (t1-t3))
   l2 = (tt-t1) * (tt-t3) / ((t2-t1) * (t2-t3))
   l3 = (tt-t1) * (tt-t2) / ((t3-t1) * (t3-t2))

   !--- Associated value
   pp = y1 * l1 + y2 * l2 + y3 * l3

END SUBROUTINE max_ht

!*****************
SUBROUTINE extrema
!*****************
!
! Find max and min of all species by the same algorithm than max_ht
! Keep only the last value
!-------------------------

   USE Chem_Aux

   IMPLICIT NONE

   INTEGER :: i
   REAL (KIND=dp)                             :: t1, t2, t3
   REAL (KIND=dp)                             :: y1, y2, y3
   REAL (KIND=dp)                             :: l1, l2, l3

   !--- Find extrema of species
   DO i = 1, nspec
      IF (abc2(i) > abc1(i) .AND. abc2(i) > abc3(i)) THEN     ! Max
         t1 = tcur(1)
         t2 = tcur(2)
         t3 = tcur(3)
         y1 = abc1(i)
         y2 = abc2(i)
         y3 = abc3(i)

         ttmax(i) = 0.5_dp * (y1*(t2**2-t3**2) - y2*(t1**2-t3**2) + y3*(t1**2-t2**2)) &
                  / (y1*(t2-t3) - y2*(t1-t3) + y3*(t1-t2))

         l1 = (ttmax(i)-t2) * (ttmax(i)-t3) / ((t1-t2) * (t1-t3))
         l2 = (ttmax(i)-t1) * (ttmax(i)-t3) / ((t2-t1) * (t2-t3))
         l3 = (ttmax(i)-t1) * (ttmax(i)-t2) / ((t3-t1) * (t3-t2))

         abmax(i) = y1 * l1 + y2 * l2 + y3 * l3
      ENDIF
      IF (abc2(i) < abc1(i) .AND. abc2(i) < abc3(i)) THEN     ! Min
         t1 = tcur(1)
         t2 = tcur(2)
         t3 = tcur(3)
         y1 = abc1(i)
         y2 = abc2(i)
         y3 = abc3(i)

         ttmin(i) = 0.5_dp * (y1*(t2**2-t3**2) - y2*(t1**2-t3**2) + y3*(t1**2-t2**2)) &
                  / (y1*(t2-t3) - y2*(t1-t3) + y3*(t1-t2))

         l1 = (ttmin(i)-t2) * (ttmin(i)-t3) / ((t1-t2) * (t1-t3))
         l2 = (ttmin(i)-t1) * (ttmin(i)-t3) / ((t2-t1) * (t2-t3))
         l3 = (ttmin(i)-t1) * (ttmin(i)-t2) / ((t3-t1) * (t3-t2))

         abmin(i) = y1 * l1 + y2 * l2 + y3 * l3
      ENDIF
   ENDDO

   !--- Find extrema of temperature
   IF (tecur(2) > tecur(1) .AND. tecur(2) > tecur(3)) THEN
      t1 = tcur(1)
      t2 = tcur(2)
      t3 = tcur(3)
      y1 = tecur(1)
      y2 = tecur(2)
      y3 = tecur(3)

      ttemax = 0.5_dp * (y1*(t2**2-t3**2) - y2*(t1**2-t3**2) + y3*(t1**2-t2**2)) &
               / (y1*(t2-t3) - y2*(t1-t3) + y3*(t1-t2))

      l1 = (ttemax-t2) * (ttemax-t3) / ((t1-t2) * (t1-t3))
      l2 = (ttemax-t1) * (ttemax-t3) / ((t2-t1) * (t2-t3))
      l3 = (ttemax-t1) * (ttemax-t2) / ((t3-t1) * (t3-t2))

      temax = y1 * l1 + y2 * l2 + y3 * l3
   ENDIF
   IF (tecur(2) < tecur(1) .AND. tecur(2) < tecur(3)) THEN
      t1 = tcur(1)
      t2 = tcur(2)
      t3 = tcur(3)
      y1 = tecur(1)
      y2 = tecur(2)
      y3 = tecur(3)

      ttemin = 0.5_dp * (y1*(t2**2-t3**2) - y2*(t1**2-t3**2) + y3*(t1**2-t2**2)) &
               / (y1*(t2-t3) - y2*(t1-t3) + y3*(t1-t2))

      l1 = (ttemin-t2) * (ttemin-t3) / ((t1-t2) * (t1-t3))
      l2 = (ttemin-t1) * (ttemin-t3) / ((t2-t1) * (t2-t3))
      l3 = (ttemin-t1) * (ttemin-t2) / ((t3-t1) * (t3-t2))

      temin = y1 * l1 + y2 * l2 + y3 * l3
   ENDIF

END SUBROUTINE extrema

REAL (KIND=dp) FUNCTION Orth_H2 (tt)

   USE Chem_Aux

   IMPLICIT NONE

   REAL (KIND=dp), INTENT (IN) :: tt
   REAL (KIND=dp)              :: Z_part

   Z_part = 1.0_dp &
          +  9.0_dp * EXP(-170.476_dp / tt) &
          +  5.0_dp * EXP(-509.864_dp / tt) &
          + 21.0_dp * EXP(-1015.08_dp / tt) &
          +  9.0_dp * EXP(-1681.64_dp / tt) &
          + 33.0_dp * EXP(-2503.74_dp / tt) &
          + 13.0_dp * EXP(-3474.50_dp / tt) &
          + 45.0_dp * EXP(-4586.06_dp / tt)

   Orth_H2 = (Z_part - 1.0_dp) / Z_part

END FUNCTION Orth_H2

END MODULE Chem_Sub
