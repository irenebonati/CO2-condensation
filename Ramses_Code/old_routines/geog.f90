      SUBROUTINE GEOG(igeog, midlatrad, midlat, oceandata, coastlat, focean, latrad, lat, ocean, nbelts)

	   
!----  SET UP GEOGRAPHY--------------------------- 

! This subroutine calculates the oceanic/zone fractions for 5 geography types (present geography, polar supercontinent, equatorial supercontinent, 100% oceanic, and 100% land)

integer ::  i,j,k,  igeog, nbelts, oceandata
real , parameter :: pi = 3.14159 
real :: latrad,  lat, midlatrad,  midlat, x,  coastlat, ocean,focean


dimension :: latrad(nbelts + 1), lat(nbelts + 1), midlatrad(nbelts), &  
       midlat(nbelts), x(nbelts+1), focean(nbelts)





!dimension :: latrad(nbelts + 1), lat(nbelts + 1), midlatrad(nbelts), &  
!       midlat(nbelts), x(nbelts+1)

! SO I  WANT AN INPUT FILE THAT TAKES IN IGEOG VALUE TO PICK TERRAIN  TYPE



      if (igeog.eq.1)then
!  PRESENT GEOGRAPHY from Sellers (1965).. Read ocean fraction  at each latitudinal band	
      rewind(oceandata)  ! goes to beginning of ocean file
      coastlat = 0.
      do k = 1, nbelts
      read(oceandata,*) focean(k)
      focean(k) = focean(k)*ocean/0.7
!	  write(60,*) focean(k)
      enddo
	  
      elseif(igeog.eq.2)then
! SOUTH POLAR SUPERCONTINENT
 coastlat = asin(1-2*ocean)*180/pi	! below this latitude, all coast (in degrees)  
	  
      do k  = 1, nbelts
	  if (lat(k).le.coastlat) then
	  focean(k) = 0.
	  
	  elseif (lat(k).gt.coastlat)then
     focean(k)= (sin(latrad(k)) - sin(coastlat*pi/180.))/(sin(latrad(k))-sin(latrad(k)-pi/nbelts))
	 else
	 focean(k) = 1.
	 endif
      enddo  ! ends south polar supercontinent
	  
      elseif(igeog.eq.3)then
! EQUATORIAL SUPERCONTINENT
 coastlat = asin(1-ocean)*180/pi
       do k = nbelts/2 +1, nbelts
       if (lat(k).le. coastlat)then
	   focean(k) =0.
	   focean(nbelts-k+1)=0.
	   
	   elseif((lat(k).gt.coastlat))then
	   focean(k) = (sin(lat(k))-sin(coastlat*pi/180.))/(sin(latrad(k))-sin(latrad(k)-pi/nbelts))
	   focean(nbelts-k+1) = focean(k)
	   else 
	    focean(k) = 1.
        focean(nbelts-k+1) = 1.
       endif
       enddo	

      elseif(igeog.eq.4)then   
! 100% WATER-COVERED

       do k = 1, nbelts
	   focean(k) = 1.
       enddo ! ends water-covered

      elseif(igeog.eq.5)then	 
! 100% LAND-COVERED

       do k = 1, nbelts
	   focean(k) = 0.
       enddo	  ! ends land-covered
	  
      endif  ! entire igeog if statements
	  
      END! END ROUTINE