! JLB - June 2020
! Some auxiliary routines and global variables for the code Chem_Evol

MODULE Chem_Aux

   IMPLICIT NONE

   INTEGER,        PARAMETER, PUBLIC  :: dp    = SELECTED_REAL_KIND (15,307)
   !INTEGER,       PARAMETER, PUBLIC  :: qp    = SELECTED_REAL_KIND (33, 4931)  ! For quadruple precision
   REAL (KIND=dp), PARAMETER, PUBLIC  :: xpi   = 3.1415926535897932384_dp
   REAL (KIND=dp), PARAMETER, PUBLIC  :: TWOPI = 6.2831853071795864769_dp

   PRIVATE

   PUBLIC :: ii2c, SFFTEU, sortwri

CONTAINS

! Used to sort eigenvalues in decreasing order
! No need for a sophisticated algorithm here
! But we need the indexes to apply this sort to several variables
!%%%%%%%%%%%%%%%%%%%%%%%%%%%
SUBROUTINE sortwri (wr, rnk)
!%%%%%%%%%%%%%%%%%%%%%%%%%%%

   IMPLICIT NONE

   REAL (KIND=dp), INTENT (IN), DIMENSION (:) :: wr           ! Real part of eigenvalues
   INTEGER, INTENT (OUT), DIMENSION (:)       :: rnk          ! Sort index

   REAL (KIND=dp), DIMENSION (:), ALLOCATABLE :: ww           ! Auxiliary array to be sorted
   INTEGER                                    :: i, j, nn
   REAL (KIND=dp)                             :: xx
   INTEGER                                    :: ll

   ! We do not want to change wr. So a copy ww is sorted in place
   ! The sorted rank is returned in rnk
   nn = SIZE(wr)
   ALLOCATE (ww(nn))
   ww  = wr
   rnk = [ (i, i = 1, nn) ]

   DO i = 1, nn-1
      DO j = i+1, nn
         IF (ww(i) < ww(j)) THEN ! Use a simple nn**2 algorithm since nn is small
            ll     = rnk(i)
            rnk(i) = rnk(j)
            rnk(j) = ll
            xx     = ww(i)
            ww(i)  = ww(j)
            ww(j)  = xx
         ENDIF
      ENDDO
   ENDDO

   DEALLOCATE (ww)

END SUBROUTINE sortwri

! Convert integers to characters to build file names
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CHARACTER (LEN=4) FUNCTION ii2c(ii)
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   IMPLICIT NONE
   INTEGER, INTENT (IN) :: ii
   INTEGER              :: icm, icc, icd, icu
   INTEGER              :: jj

   icm = 48 + ii/1000
   IF (icm == 48) THEN
      icm = 32
   ENDIF
   jj  = ii - 1000 * (ii/1000)
   icc = 48 + jj/100
   IF (icm == 32 .AND. icc == 48) THEN
      icc = 32
   ENDIF
   jj  = jj - 100 * (jj/100)
   icd = 48 + jj/10
   jj  = jj - 10 * (jj/10)
   icu = 48 + jj
   ii2c = CHAR(icm)//CHAR(icc)//CHAR(icd)//CHAR(icu)

END FUNCTION ii2c

!-------------------------------------------------------------c
!                                                             c
!  Subroutine sffteu( x, y, n, m, ityp )                      c
!                                                             c
!  This routine is a slight modification of a complex split   c
!  radix FFT routine presented by C.S. Burrus.  The original  c
!  program header is shown below.                             c
!                                                             c
!  Arguments:                                                 c
!     x - real array containing real parts of transform       c
!              sequence (in/out)                              c
!     y - real array containing imag parts of transform       c
!              sequence (in/out)                              c
!     n - integer length of transform (in)                    c
!     m - integer such that n = 2**m  (in)                    c
!     ityp - integer job specifier (in)                       c
!              ityp .ne. -1 --> foward transform              c
!              ityp .eq. -1 --> backward transform            c
!                                                             c
!  The forward transform computes                             c
!     X(k) = sum_{j=0}^{N-1} x(j)*exp(-2ijk*pi/N)             c
!                                                             c
!  The backward transform computes                            c
!     x(j) = (1/N) * sum_{k=0}^{N-1} X(k)*exp(2ijk*pi/N)      c
!                                                             c
!                                                             c
!  Requires standard FORTRAN functions - sin, cos             c
!                                                             c
!  Steve Kifowit, 9 July 1997                                 c
!                                                             c
!-------------------------------------------------------------C
!  A Duhamel-Hollman Split-Radix DIF FFT                      C
!  Reference:  Electronics Letters, January 5, 1984           C
!  Complex input and output in data arrays X and Y            C
!  Length is N = 2**M                                         C
!                                                             C
!  C.S. Burrus          Rice University         Dec 1984      C
!-------------------------------------------------------------C
!
      SUBROUTINE SFFTEU( X, Y, N, M, ITYP )
      INTEGER  N, M, ITYP
      REAL (KIND=dp) ::  X(*), Y(*)
      INTEGER  I, J, K, N1, N2, N4, IS, ID, I0, I1, I2, I3
      REAL (KIND=dp) ::  E, A, A3, CC1, SS1, CC3, SS3
      REAL (KIND=dp) ::  R1, R2, S1, S2, S3, XT
!
      IF ( N .EQ. 1 ) RETURN
!
      IF ( ITYP .EQ. -1 ) THEN
         DO 1, I = 1, N
            Y(I) = - Y(I)
 1       CONTINUE
      ENDIF
!
      N2 = 2 * N
      DO 10, K = 1, M-1
         N2 = N2 / 2
         N4 = N2 / 4
         E = TWOPI / N2
         A = 0.0
         DO 20, J = 1, N4
            A3 = 3 * A
            CC1 = COS( A )
            SS1 = SIN( A )
            CC3 = COS( A3 )
            SS3 = SIN( A3 )
            A = J * E
            IS = J
            ID = 2 * N2
 40         DO 30, I0 = IS, N-1, ID
               I1 = I0 + N4
               I2 = I1 + N4
               I3 = I2 + N4
               R1 = X(I0) - X(I2)
               X(I0) = X(I0) + X(I2)
               R2 = X(I1) - X(I3)
               X(I1) = X(I1) + X(I3)
               S1 = Y(I0) - Y(I2)
               Y(I0) = Y(I0) + Y(I2)
               S2 = Y(I1) - Y(I3)
               Y(I1) = Y(I1) + Y(I3)
               S3 = R1 - S2
               R1 = R1 + S2
               S2 = R2 - S1
               R2 = R2 + S1
               X(I2) = R1 * CC1 - S2 * SS1
               Y(I2) = - S2 * CC1 - R1 * SS1
               X(I3) = S3 * CC3 + R2 * SS3
               Y(I3) = R2 * CC3 - S3 * SS3
 30         CONTINUE
            IS = 2 * ID - N2 + J
            ID = 4 * ID
            IF ( IS .LT. N ) GOTO 40
 20      CONTINUE
 10   CONTINUE
!
!--------LAST STAGE, LENGTH-2 BUTTERFLY ----------------------C
!
      IS = 1
      ID = 4
 50   DO 60, I0 = IS, N, ID
         I1 = I0 + 1
         R1 = X(I0)
         X(I0) = R1 + X(I1)
         X(I1) = R1 - X(I1)
         R1 = Y(I0)
         Y(I0) = R1 + Y(I1)
         Y(I1) = R1 - Y(I1)
 60   CONTINUE
      IS = 2 * ID - 1
      ID = 4 * ID
      IF ( IS .LT. N ) GOTO 50
!
!-------BIT REVERSE COUNTER-----------------------------------C
!
 100  J = 1
      N1 = N - 1
      DO 104, I = 1, N1
         IF ( I .GE. J ) GOTO 101
         XT = X(J)
         X(J) = X(I)
         X(I) = XT
         XT = Y(J)
         Y(J) = Y(I)
         Y(I) = XT
 101     K = N / 2
 102     IF ( K .GE. J ) GOTO 103
         J = J - K
         K = K / 2
         GOTO 102
 103     J = J + K
 104  CONTINUE
!
      IF ( ITYP .EQ. -1 ) THEN
         DO 2, I = 1, N
            X(I) = X(I) / N
            Y(I) = - Y(I) / N
 2       CONTINUE
      ENDIF
!
      RETURN
!
! ... End of subroutine SFFTEU ...
!
      END SUBROUTINE sffteu

END MODULE Chem_Aux
