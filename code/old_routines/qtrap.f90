      SUBROUTINE qtrap(sumsteps,a,b,s,latrad,ob) 
      INTEGER JMAX 
      REAL :: a,b,s,EPS,latrad,ob 
      PARAMETER (EPS=1.e-6, JMAX=20) 
      INTEGER :: j 
      REAL olds,sumsteps  ! GOT RID of this weird extra OLDS???
	  olds = 1.e-30 ! initial value of s?
	  
	  
!	  if(sumsteps.eq.1)write(60,*) olds, a, b, s, latrad, ob
	  
      do  j=1,JMAX 
         call trapzd(a,b,s,j,latrad,ob) 
         if (j.gt.5) then 
            if (abs(s-olds).lt.EPS*abs(olds).or. &
            (s.eq.0..and.olds.eq.0.)) return 
         endif 
         olds=s 
      enddo 
      pause 'too many steps in qtrap'
      END !ends subroutine
!------------------------------------------------------------------
!     Subroutine for numerical integration from Numerical Recipes
!------------------------------------------------------------------