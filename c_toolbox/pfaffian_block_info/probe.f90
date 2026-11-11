! Optional standalone C-reference-library check for issue #731.
! Links the unmodified vendored mVMC 1.4.0 PfaPack Fortran implementation.
program probe
  implicit none
  integer :: n, i, info
  integer :: pivots(8)
  double precision :: a(8,8)
  print '(a)', 'n,connected,mode,info,analytic_pfaffian'
  do n = 2, 8, 2
    do i = 0, 1
      call run_case(n, i, 'N')
      call run_case(n, i, 'P')
    end do
  end do
contains
  subroutine run_case(n, connected, mode)
    integer, intent(in) :: n, connected
    character, intent(in) :: mode
    integer :: j
    a = 0.0d0
    do j = 1, n-1, 2
      a(j,j+1) = 2.0d0
      a(j+1,j) = -2.0d0
    end do
    if (connected == 1) then
      do j = 2, n-2, 2
        a(j,j+1) = 1.0d0
        a(j+1,j) = -1.0d0
      end do
    end if
    call dsktf2('U', mode, n, a, 8, pivots, info)
    print '(i0,a,i0,a,a,a,i0,a,f8.1)', n, ',', connected, ',', mode, ',', info, ',', 2.0d0**(n/2)
  end subroutine
end program
