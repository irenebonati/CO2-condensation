      SUBROUTINE trapzd(a,b,s,n,latrad,ob) 
      INTEGER n 
      REAL :: a,b,s,insolation1, insolation2, insolation3, latrad,ob
!      EXTERNAL insolation
      INTEGER it,j 
      REAL :: del,sumi,tnm,x 
      if (n.eq.1) then 
	  
	     insolation1 = SQRT(1 - (sin(latrad)*cos(ob) - cos(latrad)*sin(ob)*sin(a))**2)
		 insolation2 = SQRT(1 - (sin(latrad)*cos(ob) - cos(latrad)*sin(ob)*sin(b))**2)
         s=0.5*(b-a)*(insolation1 +insolation2) 
      else 
         it=2**(n-2) 
         tnm=it 
         del=(b-a)/tnm 
         x=a+0.5*del 
         sumi=0. 
         do j=1,it 
		    insolation3 = SQRT(1 - (sin(latrad)*cos(ob) - cos(latrad)*sin(ob)*sin(x))**2)
            sumi=sumi+insolation3
            x=x+del 
         enddo
         s=0.5*(s+(b-a)*sumi/tnm)         
      endif 
      return 
      END ! ends subroutine
!------------------------------------------------------------------
!     Subroutine for numerical integration from Numerical Recipes
!------------------------------------------------------------------
! Plus integrand for insolation calculation comes from Ward (1974)
