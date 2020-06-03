!---------------------------------------------------------------------c
      subroutine  keplereqn(m,e,x)
!---------------------------------------------------------------------c
!  This subroutine is used to find a solution to Keplers' equation,
!  namely M = E - eSin(E), where M is the mean anomaly, E is the
!  eccentric anomaly, and e is the eccentricity of the orbit.
!  Input is M and e, and output is E. The chosen convergence is quartic 
!  as outlined in Danby (1988), pp. 149-54.  coded from Danby (5-11-95)
!
      implicit  integer(n), real*8(a-m,o-z)
	  integer :: j
      real, parameter :: pi=3.141592653589793
      real :: ms
!
!  initial guess is x = M + k*e with k = 0.85
      k = 0.85
!  bound on the permissible error for the answer
      del = 1.e-13
!
      ms = m - int(m/(2*pi))*(2*pi)
      sigma = sign(1.,sin(ms))
      x = ms + sigma*k*e
      nc = 0
	  
       do j = 1,11  ! do a maximum of 11 times c-rr
! 100  f = x - e*sin(x) - ms     ! REPLACED THIS GOTO LOGIC WITH A DO WHILE!!!

      f = x - e*sin(x) - ms !c-rr
!  check for convergence
!      if (abs(f) .lt. del) goto 200
!      write(60,*)  nc,x,f,dx3  !**uncomment this to check for convergence
      if (abs(f) .lt. del) exit  !c-rr
      fp = 1. - e*cos(x)
      fpp = e*sin(x)
      fppp = e*cos(x)
      dx1 = -f/fp                 ! Newton's Method (quadratic convergence)
      dx2 = -f/(fp + dx1*fpp/2.)  ! Halley's Method (cubic convergence)
      dx3 = -f/(fp + dx2*fpp/2. + dx2**2*fppp/6.) ! (quartic convergence)
      x = x + dx3
!      write(60,*)  nc,x,f,dx3  !**uncomment this to check for convergence
      nc = nc + 1

	  
!  stop calculation if number of iterations is too large
      if (nc .gt. 10) then
         write(*,*) 'no solution to keplers equation'
         stop
      end if
	  
      enddo  ! ends do loop
!      goto 100
! 200  return
      end !ends subroutine