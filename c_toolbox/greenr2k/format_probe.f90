! List-directed and edit-descriptor output probe for gfortran (issue #351).
! Reads one 64-bit hexadecimal pattern per line from stdin; for the REAL(8) value
! x and y = REAL(x) (REAL(4)) prints one output record per format used by
! greenr2k.F90, as "<hex> <tag> [<record text>]".  Each record is written to an
! internal unit, so every blank (including trailing ones) is delimited by the
! brackets and the exact record length is known through the SIZE of the write.
PROGRAM probe
  IMPLICIT NONE
  CHARACTER(32) :: line
  INTEGER(8) :: bits
  REAL(8) :: x
  REAL(4) :: y
  CHARACTER(120) :: s
  INTEGER :: ios
  DO
     READ(*,'(a)',IOSTAT=ios) line
     IF (ios /= 0) EXIT
     READ(line,'(z16)') bits
     x = TRANSFER(bits, x)
     y = REAL(x)
     CALL emit("r8 ")
     CALL emit("r4 ")
     CALL emit("e15")
     CALL emit("f15")
     CALL emit("f10")
     CALL emit("f72")
  END DO
CONTAINS
  SUBROUTINE emit(tag)
    CHARACTER(3) :: tag
    s = ""
    SELECT CASE (tag)
    CASE ("r8 ")
       WRITE(s,*) x
    CASE ("r4 ")
       WRITE(s,*) y
    CASE ("e15")
       WRITE(s,'(e15.5)') x
    CASE ("f15")
       WRITE(s,'(f15.10)') x
    CASE ("f10")
       WRITE(s,'(f10.5)') x
    CASE ("f72")
       WRITE(s,'(f7.2)') x
    END SELECT
    ! Record length: list-directed writes pad the internal record with blanks to
    ! len(s); the formats of greenr2k only ever pad trailing blanks in the real
    ! list-directed item, which the probe reports with a fixed 30 columns.
    WRITE(*,'(a,1x,a,1x,a,a,a)') TRIM(line), tag, "[", s(1:30), "]"
  END SUBROUTINE emit
END PROGRAM probe
