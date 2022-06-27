! JLB - June 2020
! This utility tool reads an output file from the code Chem_Evol
! and allows to analyse the chemical reactions set.
! Che_Evol should first run with the flag F_CH_A = 1

PROGRAM Chem_Analys

   USE Chem_Aux
   USE Chem_Sub

   IMPLICIT NONE

   INTEGER                                    :: i, j
   INTEGER                                    :: ir1, ir2
   INTEGER                                    :: ip1, ip2, ip3, ip4
   CHARACTER (LEN=8)                          :: the_spec
   INTEGER                                    :: i_spec
   REAL (KIND=dp)                             :: rate
   REAL (KIND=dp)                             :: thresh
   INTEGER                                    :: nf, nd
   REAL (KIND=dp)                             :: kft, kdt
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE :: kf, kd
   INTEGER       , DIMENSION (:), ALLOCATABLE :: rf, rd
   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE :: dy

   !--- Read state file (*.cha)
   PRINT *, "# Enter file name"
   READ(5,'(a50)') fichi
   PRINT *, "# Opening ", fichi
   !fichi = "Base_01.cha"
   OPEN (iwrtc, FILE=fichi, STATUS="old")
   !--- Model description
   READ (iwrtc,*) densh, tempr, zeta, rop
   !--- Time
   READ (iwrtc,*) ts
   !--- Dimensions
   READ (iwrtc,*) nspec, neqt
   neqev = nmhd + nspec

   ALLOCATE (speci(0:nspec+npart))
   ALLOCATE (abev(0:nspec+npart))
   ALLOCATE (react(neqt,6))
   ALLOCATE (itype(neqt))
   ALLOCATE (xkrate(neqt))
   ALLOCATE (dy(neqev))

   !--- List of species
   READ (iwrtc,'(a8)') (speci(i), i=0,nspec+npart)
   !--- Specific species indexes
   READ (iwrtc,*) i_h, i_c, i_n, i_o, i_he, i_d, i_c13, i_n15, i_o18, i_ar &
                , i_f, i_na, i_mg, i_gr, i_si, i_ne, i_s, i_cl, i_ca, i_fe &
                , i_h2, i_hd, i_c2, i_n2, i_h2o, i_oh, i_ch, i_co, i_hp, i_cp &
                , i_sp, i_np, i_nhp, i_fep, i_h2dp, i_c13p, i_n15p

   !--- Right hand side
   READ (iwrtc,*) (dy(i), i=1,neqev)
   !--- Abundances
   READ (iwrtc,*) (abev(i), i=1,nspec+npart)
   !--- reactions and rates
   READ (iwrtc,*) (react(i,1:6), itype(i), xkrate(i), i=1,neqt)
   CLOSE (iwrtc)
   ALLOCATE (kf(neqt))
   ALLOCATE (kd(neqt))
   ALLOCATE (rf(neqt))
   ALLOCATE (rd(neqt))

   !--- Start output
   PRINT *, " "
   PRINT *, "# Chemical analysis of: ", fichi
   WRITE (6,'(" #    nH = ",1pe15.6," (cm-3),    T = ",e15.6," (K)")') densh, tempr
   WRITE (6,'(" #  zeta = ",1pe15.6,"       ,  O/P = ",e15.6)') zeta, rop
   PRINT *, " "
   PRINT *, "# Enter threshold (typical value: 100)"
   READ  *, thresh
   PRINT *, "# Keep reactions above ", 1.0_dp / thresh
   PRINT *, " "

   !--- Start analysis here
   DO
       PRINT *, "# Which species do you want? (End: 0)"
       READ *, the_spec
       IF (the_spec == "0") EXIT
       DO i = 1, nspec
          IF (speci(i) == the_spec) THEN
             i_spec = i
             EXIT
          ENDIF
       ENDDO
       IF (i == nspec+1) THEN
          PRINT *, the_spec, " is not included in:", fichi
          PRINT *, " Please chose another species."
          CYCLE
       ENDIF
       WRITE (6,'(" #",i4," n(",a,") = ",1pe15.6," (cm-3)")') i_spec, TRIM(ADJUSTL(speci(i_spec))), abev(i_spec)
       PRINT *, " "

       !--- Compute chemical rates
       nf = 0
       nd = 0
       kf = 0
       kd = 0
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
                rate = xkrate(i) * abev(ir1) * abev(ir2) / abev(i_h)
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

       CALL permut_r

       kft = SUM(kf(1:nf))
       kdt = SUM(kd(1:nd))
       WRITE (6,'(" # Formation/destruction of: ",a,": ",1p,4e16.7," (cm-3 s-1)",e16.7,2i4)') &
             TRIM(ADJUSTL(speci(i_spec))), kft, kdt, kft - kdt, dy(nmhd+i_spec) &
           , ABS(ABS(kft - kdt) - ABS(dy(nmhd+i_spec))) /  ABS(dy(nmhd+i_spec)) &
           , nf, nd
       WRITE (6,'(" # time = ",1pe15.6, " Myr",/)') ts

       WRITE (6,'(" # Formation of: ",a," :",1pe15.6," (cm-3 s-1)")') TRIM(ADJUSTL(speci(i_spec))), kft
       DO i = 1, nf
          IF (kft / kf(i) > thresh) EXIT
          ir1 = react(rf(i),1)
          ir2 = react(rf(i),2)
          ip1 = react(rf(i),3)
          ip2 = react(rf(i),4)
          ip3 = react(rf(i),5)
          ip4 = react(rf(i),6)
          WRITE (6,'(2i6,2x,a8," + ",a8," -> ",a8," + ",a8," + ",a8," + ",a8,1pe15.6," (cm-3 s-1)" &
                ,0pf8.2, "%",1pe15.6," (s-1)")') &
                i, itype(rf(i)) &
              , speci(ir1), speci(ir2), speci(ip1), speci(ip2), speci(ip3), speci(ip4) &
              , kf(i), 100.0_dp * kf(i) / kft, kf(i) / abev(i_spec)
       ENDDO
       PRINT *, " "

       WRITE (6,'(" # Destruction of: ",a," :",1pe15.6," (cm-3 s-1)")') TRIM(ADJUSTL(speci(i_spec))), kdt
       DO i = 1, nd
          IF (kdt / kd(i) > thresh) EXIT
          ir1 = react(rd(i),1)
          ir2 = react(rd(i),2)
          ip1 = react(rd(i),3)
          ip2 = react(rd(i),4)
          ip3 = react(rd(i),5)
          ip4 = react(rd(i),6)
          WRITE (6,'(2i6,2x,a8," + ",a8," -> ",a8," + ",a8," + ",a8," + ",a8,1pe15.6," (cm-3 s-1)" &
                ,0pf8.2, "%",1pe15.6," (s-1)")') &
                i, itype(rd(i)) &
              , speci(ir1), speci(ir2), speci(ip1), speci(ip2), speci(ip3), speci(ip4) &
              , kd(i), 100.0_dp * kd(i) / kdt, kd(i) / abev(i_spec)
       ENDDO
       PRINT *, " "

   ENDDO
   PRINT *, "# End! "
   PRINT *, " "

   DEALLOCATE (speci)
   DEALLOCATE (abev)
   DEALLOCATE (react)
   DEALLOCATE (itype)
   DEALLOCATE (xkrate)
   DEALLOCATE (kf)
   DEALLOCATE (kd)
   DEALLOCATE (rf)
   DEALLOCATE (rd)

CONTAINS

SUBROUTINE permut_r

   IMPLICIT NONE

   REAL (KIND=dp) :: r1
   INTEGER        :: i1

   DO i = 1, nf-1
      DO j = i, nf
         IF (kf(j) > kf(i)) THEN
            r1 = kf(i)
            kf(i) = kf(j)
            kf(j) = r1
            i1 = rf(i)
            rf(i) = rf(j)
            rf(j) = i1
         ENDIF
      ENDDO
   ENDDO

   DO i = 1, nd-1
      DO j = i, nd
         IF (kd(j) > kd(i)) THEN
            r1 = kd(i)
            kd(i) = kd(j)
            kd(j) = r1
            i1 = rd(i)
            rd(i) = rd(j)
            rd(j) = i1
         ENDIF
      ENDDO
   ENDDO

END SUBROUTINE permut_r

END PROGRAM Chem_Analys
