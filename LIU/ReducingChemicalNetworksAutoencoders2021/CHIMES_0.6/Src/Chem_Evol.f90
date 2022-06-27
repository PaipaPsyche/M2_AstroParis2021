PROGRAM Chem_Evol
!---------------------------------------------------------------------!
!      Main program of evolution code CHIMES                          !
!      *************************************                          !
!                                                                     !
!      Jacques Le Bourlot & Evelyne Roueff - 2020                     !
!                                                                     !
!      Derived and updated from an old (about 1990)                   !
!      version of Durham MHD shock code                               !
!      Monofluid version only                                         !
!      By David Flower and Guillaume Pinault des Forets               !
!                                                                     !
!---------------------------------------------------------------------!

   USE DVODE_F90_M
   USE Chem_Aux
   USE Chem_Sub

   IMPLICIT NONE

   ! Variables for DVODE
   REAL (KIND=dp)                             :: t0, tin, tout
   REAL (KIND=dp)                             :: dtout
   TYPE (VODE_OPTS)                           :: OPTIONS
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE :: ATOL
   REAL (KIND=dp)                             :: RTOL
   INTEGER                                    :: ITASK
   INTEGER                                    :: ISTATE
   !REAL (KIND=dp), DIMENSION (22)            :: RSTATS
   !INTEGER       , DIMENSION (31)            :: ISTATS

   !--- Variables for SFFTEU
   INTEGER, PARAMETER                         :: m_fft = 6
   REAL (KIND=dp), DIMENSION (nfft, m_fft)    :: s_fft
   REAL (KIND=dp), DIMENSION (nfft)           :: x_fft
   REAL (KIND=dp), DIMENSION (nfft)           :: y_fft
   REAL (KIND=dp)                             :: dnu
   INTEGER                                    :: i_fft
   INTEGER                                    :: ITYP = 1

   INTEGER                                    :: i, j, k
   INTEGER                                    :: kmax = 2000000
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE :: dy
   REAL (KIND=dp)                             :: n0, z0
   REAL (KIND=dp)                             :: dt1, dt2
   INTEGER                                    :: nst, nt
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE :: tt, pp

  !--- Debug Analys
  INTEGER                                    :: ir1, ir2
  INTEGER                                    :: ip1, ip2, ip3, ip4
  INTEGER                                    :: i_spec
  REAL (KIND=dp)                             :: rate
  INTEGER                                    :: nf, nd
  REAL (KIND=dp)                             :: kft, kdt
  REAL (KIND=dp), DIMENSION (:), ALLOCATABLE :: kf, kd
  INTEGER       , DIMENSION (:), ALLOCATABLE :: rf, rd

   !-----------------------------------------------------------

   !--- Read input parameters
   CALL init

   !--- This file keeps successive times of maxima of H
   !--- Useful to estimate a relaxation time
   fichi = TRIM(ADJUSTL(outdir))//TRIM(ADJUSTL(rootnm)) // ".max"
   OPEN  (iwrtm, FILE = fichi, STATUS='unknown')
   WRITE (iwrtm,'("# tt H-max")') 

   !--- Allow for some more extrema during the transitory phase
   IF (.NOT. ALLOCATED(tt)) THEN
      nst = INT(1.3_dp * tend / h0)
      ALLOCATE (tt(nst), pp(nst))
   ENDIF

   nt = 0

   !--- Read species, chemical reactions and initial abundances
   CALL lectur

   ! Total number of evolution equations
   neqev = nx2
   ALLOCATE (dy(neqev))
  !--- Debug Analys
  ALLOCATE (kf(neqt))
  ALLOCATE (kd(neqt))
  ALLOCATE (rf(neqt))
  ALLOCATE (rd(neqt))
   idbg = 0

   !--- Prepare integration

   !--- First MHD variables, then chemical abundances
   f0(1)       = SUM(abin(1:nspec))
   f0(2)       = tempr
   f0(nx1:nx2) = abin(1:nspec)

   !--- convert to logarithmic scale
   f0(1:neqev) = LOG(f0(1:neqev))

   !--- Integrator arguments
   ALLOCATE (ATOL(neqev))
   ATOL        = 1.0e-20_dp
   RTOL        = 1.0e-12_dp
   ITASK  = 1
   ISTATE = 1

   !--- Initialize integrator
   OPTIONS = SET_OPTS(DENSE_J=.TRUE., ABSERR_VECTOR=ATOL, RELERR=RTOL, &
             USER_SUPPLIED_JACOBIAN=.FALSE., MXSTEP=10000, &
             METHOD_FLAG=22)

   !--- Integration starts with a very small step size: tout = 1 s
   !--- tout increases geometrically by multiplicative steps dtout up to h0
   dtout  = 1.1_dp
   tout   = 1.0_dp / dtout
   t0     = 0.0_dp

   k = 0
   i_fft = 0
   DO WHILE (tout < tend .AND. k < kmax)
      k = k + 1

      tin  = tout
      tout = MIN(tout * dtout, tout + h0)
      tout = MIN(tout, tend)
      CALL DVODE_F90 (diffun, neqev, f0, t0, tout, ITASK, ISTATE, OPTIONS)
      !CALL GET_STATS(RSTATS,ISTATS)
      !   WRITE (iscrn,'("#",1p,3e15.6,10i8)') RSTATS(11:13), ISTATS(11:16), ISTATS(19:22)
      !IF (ISTATE == -1) THEN
      !   PRINT *, " Retry failed step"
      !   ISTATE = 2
      !ENDIF
      !stop

      !ATOL = 1.0e-6*RTOL * MAX(ATOL,f0(1:neqev))
      fy(1:neqev) = EXP(f0(1:neqev))

      xdens = fy(1)
      tempr = fy(2)
      press = xdens * tempr

      abev(1:nspec) = fy(nx1:nx2)

      !--- H and H2 density
      xnh  = abev(i_h)
      xnh2 = abev(i_h2)

      !--- Relative abondunces (wrt protons or H2)
      abrelh2(1:nspec) = abev(1:nspec) / xnh2

      !--- Convert time to Myrs
      hs = (tout - tin) / Myr
      ts = tout         / Myr
      IF (MOD(k,ks) == 0) THEN
         WRITE (iscrn,2069) k, hs, ts, tempr, abev(nspec), xnh, xnh2, orth/para
         !WRITE (iwrto,2069) k, hs, ts, tempr, abev(nspec), xnh, xnh2, orth/para

         !--- Standard outputs (is it really useful?)
         !WRITE (iwrto,2075) (abev(i), abrelh2(i), speci(i), i=1,nspec)
         !WRITE (iwrto,2120)

         !--- H2 rotational populations
         !WRITE (iwrto,2140) (abj(i), i-1, i=1,7)
         !WRITE (iwrto,2120)
      ENDIF

      !--- Evolution variables, suited for plots
      IF (MOD(k,ks) == 0) THEN
      !IF (MOD(k,ks) == 0 .AND. ts > 0.8 * tend / Myr) THEN
         WRITE (iwrtg,'(405(1pe16.8))') ts, zeta, densh, tempr, (abev(i), i=1,nspec) &
                     , bthb, ynn, bnn, bee, bchim, gamh2, gamgr, gampeg, zlambd, xlamoh &
                     , xlamco, xlah2o, xlamc, xlamcp, xlamo, xlambn, wthb, xlambe, gamrc
         CALL diffun(neqev, ts, f0, dy)
         dy(1:neqev) = dy(1:neqev) * fy(1:neqev)
         abdot(1:nspec) = dy(nx1:nx2)
         WRITE (iwrtd,'(405(1pe16.8))') ts, zeta, densh, tempr, (abdot(i), i=1,nspec) &
                     , bthb, ynn, bnn, bee, bchim, gamh2, gamgr, gampeg, zlambd, xlamoh &
                     , xlamco, xlah2o, xlamc, xlamcp, xlamo, xlambn, wthb, xlambe, gamrc
      ENDIF

      !--- Look for maximum in H evolution
      !--- Used to compute transitory time scale, and evaluate period
      !--- Keep track of extrema for Temperature and all species
      tcur(1)  = tcur(2)
      tcur(2)  = tcur(3)
      tcur(3)  = ts
      abc1     = abc2
      abc2     = abc3
      abc3     = abev(1:nspec)
      tecur(1) = tecur(2)
      tecur(2) = tecur(3)
      tecur(3) = tempr
      IF (abc2(i_h) > abc1(i_h) .AND. abc2(i_h) > abc3(i_h)) THEN
         nt = nt + 1
         CALL max_ht (tt(nt), pp(nt))
         WRITE (iwrtm,*) tt(nt), pp(nt)
      ENDIF
      CALL extrema

      !--- Prepare FFT computation
      !--- Chose here which species (default is H)
      IF (F_FFT == 1) THEN
         IF (ts > t_trans) THEN
            IF (tout-tin /= h0) THEN
               PRINT *, " Wrong sampling rate for FFT"
               PRINT *, "       use longer t_trans"
               PRINT *, "       t_trans =", t_trans
            ENDIF
            i_fft = i_fft + 1
            s_fft(i_fft,1) = abev(i_c)
            s_fft(i_fft,2) = abev(i_co)
            s_fft(i_fft,3) = abev(i_ch4)
            s_fft(i_fft,4) = abev(i_cd5p)
            s_fft(i_fft,5) = abev(i_nh3)
            s_fft(i_fft,6) = abev(i_sh)
            IF (i_fft == nfft) EXIT
         ENDIF
      ENDIF

      !--- Save state to analyse chemistry
      IF (F_CH_A == 1) THEN
         IF (ts > t_sav) THEN
            idbg = 1
            i_sav = i_sav + 1
            fichi = TRIM(ADJUSTL(outdir))//TRIM(ADJUSTL(rootnm)) &
                  // "_" // TRIM(ADJUSTL(ii2c(i_sav))) // ".cha"
            OPEN (iwrtc, FILE=fichi, STATUS="unknown")
            !--- Model description
            WRITE (iwrtc,*) densh, tempr, zeta, orth/para
            !--- Time
            WRITE (iwrtc,*) ts
            !--- Dimensions
            WRITE (iwrtc,*) nspec, neqt
            !--- List of species
            WRITE (iwrtc,'(a8)') (speci(i), i=0,nspec+npart)
            !--- Specific species indexes
            WRITE (iwrtc,*) i_h, i_c, i_n, i_o, i_he, i_d, i_c13, i_n15, i_o18, i_ar &
                          , i_f, i_na, i_mg, i_gr, i_si, i_ne, i_s, i_cl, i_ca, i_fe &
                          , i_h2, i_hd, i_c2, i_n2, i_h2o, i_oh, i_ch, i_co, i_hp, i_cp &
                          , i_sp, i_np, i_nhp, i_fep, i_h2dp, i_c13p, i_n15p

            CALL diffun (neqev, ts, f0, dy)
            dy(1:neqev) = dy(1:neqev) * fy(1:neqev)
            !--- Right hand side
            WRITE (iwrtc,*) (dy(i), i=1,neqev)
            !--- Abundances
            WRITE (iwrtc,*) (abev(i), i=1,nspec+npart)
            !--- reactions and rates
            WRITE (iwrtc,*) (react(i,1:6), itype(i), xkrate(i), i=1,neqt)
            CLOSE (iwrtc)

      !--- Debug Analys
      nf = 0
      nd = 0
      kf = 0
      kd = 0
      !i_spec = i_ch
      i_spec = nspec
      DO i = 1, neqt
         ir1 = react(i,1)
         ir2 = react(i,2)
         ip1 = react(i,3)
         ip2 = react(i,4)
         ip3 = react(i,5)
         ip4 = react(i,6)
         !--- Scale type 0 reactions (H2, HD and D2 formation on grains)
         IF (itype(i) == 0) THEN
            IF (((ir1 == i_h) .OR. (ir2 == i_h))) THEN
               rate = xkrate(i)  * abev(ir1) * abev(ir2) / abev(i_h)
            ELSE IF ((ir1 == i_d) .AND. (ir2 == i_d)) THEN
               rate = xkrate(i) * abev(ir1) * abev(ir2) / abev(i_d)
            ENDIF
         ELSE
             rate = xkrate(i) * abev(ir1) * abev(ir2)
         ENDIF

         IF (ip1 == i_spec) THEN
            nf = nf + 1
            rf(nf) = i
            kf(nf) = rate
         ENDIF
         IF (ip2 == i_spec) THEN
            nf = nf + 1
            rf(nf) = i
            kf(nf) = rate
         ENDIF
         IF (ip3 == i_spec) THEN
            nf = nf + 1
            rf(nf) = i
            kf(nf) = rate
         ENDIF
         IF (ip4 == i_spec) THEN
            nf = nf + 1
            rf(nf) = i
            kf(nf) = rate
         ENDIF
         IF (ir1 == i_spec) THEN
            nd = nd + 1
            rd(nd) = i
            kd(nd) = rate
         ENDIF
         IF (ir2 == i_spec) THEN
            nd = nd + 1
            rd(nd) = i
            kd(nd) = rate
         ENDIF
      ENDDO

      kft = SUM(kf(1:nf))
      kdt = SUM(kd(1:nd))
      WRITE (6,'(" # Formation/destruction of: ",a,": ",1p,4e16.7," (cm-3 s-1)",e16.7,2i4)') &
            TRIM(ADJUSTL(speci(i_spec))), kft, kdt, kft - kdt, dy(nmhd+i_spec) &
          , ABS(ABS(kft - kdt) - ABS(dy(nmhd+i_spec))) /  ABS(dy(nmhd+i_spec)) &
          , nf, nd

            !--- One file every 1/5 of a period
            t_sav = t_sav + 0.20_dp * t_per
            idbg = 0
         ENDIF
      ENDIF

   ENDDO

   !--- Compute here steady state, starting from current state
   CALL steady_st

   !--- Compute FFT of H evolution if required
   IF (F_FFT == 1) THEN
      fichi = TRIM(ADJUSTL(outdir))//TRIM(ADJUSTL(rootnm)) // ".fft"
      OPEN (iwrtf, FILE=fichi, STATUS="unknown")
      dnu = 1.0_dp / (REAL(nfft,dp) * h0 / Myr)
      DO j = 1, 6
         x_fft = s_fft(:,j) - SUM(s_fft(:,j)) / REAL(nfft,dp)
         y_fft = 0.0_dp
         CALL SFFTEU(x_fft, y_fft, nfft, mfft, ITYP )
         WRITE (iwrtf,'("# dnu =",1pe15.6)') dnu
         DO i = 2, nfft/2
            WRITE (iwrtf,*) i, REAL(i-1,dp) * dnu, SQRT(x_fft(i)**2+y_fft(i)**2)
         ENDDO
         WRITE (iwrtf,'(" ")')
      ENDDO
      CLOSE (iwrtf)
   ENDIF

   ! nt is the number of maxima found for H
   WRITE (iscrn,'(" ")')
   IF (nt < 3) THEN
      WRITE (iscrn,*) "    No oscillations found for H"
      WRITE (iscrn,'(a2,1p,6e15.6)') "##", densh, tempr, zeta, abev(nspec)
   ELSE
      dt1 = tt(nt-1) - tt(nt-2)
      dt2 = tt(nt) - tt(nt-1)
      IF (ABS(dt2-dt1)/(ABS(dt2)+ABS(dt1)) < 1.0e-3_dp .AND. tt(nt) > 0.5_dp * tend / Myr) THEN
         WRITE (iscrn,*) "    Oscillations found"
         WRITE (iscrn,*) "    nH   T   zeta   Per1   Per2   n(e-)"
         WRITE (iscrn,'(a2,1p,6e15.6)') "  ", densh, tempr, zeta, dt1, dt2, abev(nspec)
      ELSE
         WRITE (iscrn,*) "    Oscillations unreliable - check carefully"
         WRITE (iscrn,*) "    nH   T   zeta   Per1   Per2   n(e-)"
         WRITE (iscrn,'(a2,1p,6e15.6)') "##", densh, tempr, zeta, dt1, dt2, abev(nspec)
      ENDIF
   ENDIF

   !--- Final abundances - in cm-3 and relative to H2
   WRITE (iwrto,3999)
   DO i = 1, nspec
      WRITE (iwrto,4001) i, speci(i), abev(i), abrelh2(i), observ(i)
   ENDDO
   WRITE (iwrto,4003)

   DEALLOCATE (ATOL)
   DEALLOCATE (dy)

!----------------------------------------------------------------------
 2069 FORMAT (1x,'*',i7,1p,' dt =',e11.4,' t =',e11.4,' tempr =',e11.4,'   ne =',e11.4, &
              '   nH =',e11.4,' nh2 =',e11.4,'  o/p =',e11.4)
 2075 FORMAT (4(1x,1pd8.2,2x,d8.2,' : ',a7))
 2120 FORMAT (1x)
 2140 FORMAT (7(1pd10.2,':j=',i1,2x))
 3999 FORMAT (1x,67("-"),/, &
             6x,'espece',6x,'n(x)',11x,'n(xd)/n(xh)',3x,'n(x)/nh2',/, &
                                                       1x,67("-"),/)
 4001 FORMAT (2x, i3,2x,a7,': ',1pe14.6,' (cm-3)',4x, 8x ,4x,d8.2,2x,a23)
 4003 FORMAT (1x,67("-"),/)

END PROGRAM Chem_Evol
