       SUBROUTINE WEATHERING(area, k, nbelts, focean, fwthr, wthrate, warea, temp)

!  CARBONATE-SILICATE WEATHERING
IMPLICIT NONE

real :: fwthr, area, warea, focean,  wthrate,  temp
DIMENSION :: area(nbelts), warea(nbelts), focean(nbelts), wthrate(nbelts), temp(nbelts+1)
integer ::  nbelts , k

warea(k) = area(k)*(1-focean(k))*fwthr   ! weathering area, with weatherable fraction (fwthr)
wthrate(k) = warea(k)*(1+0.087*(temp(k)-288.) + &
  1.862e-3*(temp(k)-288.)**2)  ! weathering rate


!if(k.eq.1)write(60,*)temp(k), warea(k),  k, fwthr, focean(k), wthrate(k)	 
	 
END ! ends subrouine  
