      SUBROUTINE WRAPUP(last, k,ann_tempave, ann_albave, ann_fluxave, ann_irave, ann_wthrateave, &
	  wco2,d, pco2, nbelts, modelout,zntempmin, tcalc)
	  
	  IMPLICIT NONE
	  integer :: nbelts, last,k
	  REAL :: ann_albave, ann_fluxave, ann_irave, ann_wthrateave, wco2,d, pco2, &
	  zntempmin, ann_tempave, tcalc
	  DIMENSION :: zntempmin(nbelts)
      integer :: modelout
!  WRAP UP
!
 1000 write(*,*) 'The energy-balance calculation is complete.'
      write(modelout,1010)
 1010 format(/ 'SUMMARY')
      write(modelout,1015) 'planet average temperature = ', ann_tempave, &
     ' Kelvin'
      write(modelout,1015) 'planet average albedo = ', ann_albave
      write(modelout,1015) 'planet average insolation = ', ann_fluxave, &
      ' Watts/m^2'
      write(modelout,1015) 'planet average outgoing infrared = ',ann_irave, & 
      ' Watts/m^2'
      write(modelout,1016) 'seasonal-average weathering rate = ', &
      ann_wthrateave*wco2,' g/year'
      write(modelout,1016) 'co2 partial pressure = ', pco2, ' bars'
 1015 format(3x,a,f7.3,a)
      write(modelout,1016) 'thermal diffusion coefficient (D) = ', d, & 
      ' Watts/m^2 K'
 1016 format(3x,a,e8.3,a)
      write(modelout,1020) 'convergence to final temperature profile in ', &
      tcalc, ' seconds'
 1020 format(3x,a,e9.3,a)
 
  
       
      end   ! ENDS SUBROUTINE
