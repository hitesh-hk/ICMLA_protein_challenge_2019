C deriv.inc
C derivatives and velocities (for MD)
C
C MAXAT5 must match MAXA
C
      INTEGER MAXAT5
      PARAMETER (MAXAT5=MAX_ATOM)
      DOUBLE PRECISION DX(MAXAT5), DY(MAXAT5), DZ(MAXAT5)
      DOUBLE PRECISION XV(MAXAT5), YV(MAXAT5), ZV(MAXAT5)
C
      COMMON /DERIV/ DX, DY, DZ, XV, YV, ZV
      SAVE /DERIV/