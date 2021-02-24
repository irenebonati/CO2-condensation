! FINALLY GET HERE ONCE t.Ge.P and last.eq.1. And WRITE THE ZONAL  STATISTICS.
       SUBROUTINE ZONALEND(shtempave, nhtempave, nbelts, zntempsum, &
	   zntempave, nstep, k, icelat, iceline, nedge, lat, modelout, &
	   zndecmin, zndecmax,zntempmax, zntempmin, ann_co2cldave, igeog, &
	   focean, coastlat, ocean, seasonout, midlat,x)
	   
       IMPLICIT NONE
		  
       integer :: nbelts, k, igeog, nstep, nedge, modelout, seasonout
		  
       real*4 :: shtempave, nhtempave, zntempsum, zntempave, icelat, &
	   iceline, lat, zndecmin, zndecmax, zntempmax, zntempmin, &
	   ann_co2cldave, focean, coastlat, ocean, midlat, x
		  
		  
       dimension :: lat(0:nbelts + 1), focean(nbelts),  &
       zntempmin(nbelts), zntempmax(nbelts), zntempsum(nbelts), & 
	   zntempave(0:nbelts+1), zndecmax(nbelts), zndecmin(nbelts),& 
	   iceline(0:5), midlat(nbelts), x(0:nbelts+1)
		  		  

	     write(modelout,1130)
 1130 format(/ 'ZONAL STATISTICS')
         write(modelout,1135)
 1135 format(1x,'latitude(deg)',2x,'ave temp(K)',2x,'min temp(K)', &
      2x,'@dec(deg)',2x,'max temp(K)',2x,'@dec(deg)',2x, &
      'seas.amp(K)')

      write(modelout,1100) 
 1100 format(/ 'OUTPUT FILES')
	  
	  
! ZONAL ORBITAL AVERAGES OF LAST TIME STEP OF LAST ORBIT 
      shtempave = 0   !southern hemisphere temperature average initial value
      nhtempave = 0   ! northern hemisphere temperature average initial value
	  
!	  write(50,*) shtempave, nhtempave
	  
	  

!      zntempave(1:nbelts) = zntempsum(1:nbelts)/(nstep)   ! WHAT THE HECK?? This is why iceball is being wrong!!
!      zntempave(1:nbelts) = 0.5*(zntempmin(1:nbelts) +  zntempmax(1:nbelts)) ! alternate logic	 

! 	     write(40,*)zntempave(1:18), zntempsum(1:18), nstep	 
		 
      do  k = 1, nbelts, 1	
         zntempave(k) = zntempsum(k)/nstep 
         if (k.le.(nbelts/2)) then
           shtempave = shtempave + zntempave(k)
         elseif(k.gt.(nbelts/2))then
           nhtempave = nhtempave + zntempave(k)
         end if   
!       write(60,*)shtempave,  nhtempave, zntempave(k),k, nbelts/2		 
      enddo
	  
      shtempave = shtempave/(nbelts/2)
      nhtempave = nhtempave/(nbelts/2)
      print *, "SH/NH temperature difference = ", shtempave - nhtempave
	  
!	  write(60,*)shtempave, nhtempave
      zntempave(nbelts+1) = zntempave(nbelts)  !**for ice-line calculation

!  FIND ICE-LINES (ANNUAL-AVERAGE TEMP < 263.15K)
      nedge = 0
      do  k=1,nbelts,1
        if ((zntempave(k+1)-263.15)*(zntempave(k)-263.15) .lt. 0.) then
         icelat = lat(k) + ((lat(k+1)-lat(k))/ &
        (zntempave(k+1)-zntempave(k)))*(263.15-zntempave(k))
         nedge = nedge + 1
         iceline(nedge) = icelat

        end if
!	     write(60,*) nedge,lat(k), lat(k+1), zntempave(k+1), zntempave(k)
      enddo  ! icelines loop

      do  k=1,nbelts,1   !THIS IS THE MAIN LATITUDE TEMPERATURE OUTPUT
         write(modelout,751) lat(k),zntempave(k),zntempmin(k), &
           zndecmin(k),zntempmax(k),zndecmax(k), &
           (zntempmax(k)-zntempmin(k))/2.
      enddo   
		 
!         write(50,*)sumsteps, zndecmin(k), zndecmax(k) 		 
		   
		   
 751     format(4x,f4.0,9x,f8.3,5x,f8.3,5x,f6.2,5x,f8.3,5x,f6.2,5x,f8.3)
         write(seasonout,752) lat(k),(zntempmax(k)-zntempmin(k))/2., &
          10.8*abs(x(k)),15.5*abs(x(k)),6.2*abs(x(k))
 752     format(4x,f4.0,4x,4(3x,f8.3))
   
      
!      write(15,755)
 755  format(/ 'SURFACE DATA')
!      write(15,756)
 756  format(2x,'latitude(deg)',2x,'temp(k)',2x,'belt area', &
       2x,'weathering area',2x,'zonal weathering rate (g/yr)')
	   
      do  k = 1,nbelts,1
!         write(modelout,757) lat(k),temp(k),area(k),warea(k), &
!          wthrate(k)
 757     format(6x,f4.0,8x,f8.3,10x,f5.3,10x,f5.3,15x,e8.2)
      enddo

       write(modelout,760)
 760  format(/ 'ICE LINES (Tave = 263K)')
  
      if((nedge.eq.0).and.(zntempave(nbelts/2).le.263.)) then
      	write(modelout,*) '  planet is an ice-ball.' 
      else if((nedge.eq.0).and.(zntempave(nbelts/2).gt.263.)) then
     	write(modelout,*) '  planet is ice-free.'
      else
        do k=1,nedge,1
           write(modelout,762) iceline(k)
 762       format(2x,'ice-line latitude = ',f5.1,' degrees.')
        enddo
      end if

!  CO2 CLOUDS
      write(modelout,770)
 770  format(/ 'CO2 CLOUDS')
      write(modelout,772) ann_co2cldave
 772  format(2x,'planet average co2 cloud coverage = ',f5.3)

!  GEOGRAPHY INFORMATION 

 779  format (/ 'GEOGRAPHIC DATA')
      write(modelout,779)
      write(modelout,780) ocean
 780  format('planet ocean fraction: ',f3.1)
      write(modelout,782) coastlat
 782  format('coast latitude: ',f5.1)
      write(modelout,784) igeog
 784  format('geography: ',i1)
      write(modelout,785) nbelts
 785  format('number of belts: ',i2)
      write(modelout,786) 
 786  format(2x,'latitude of belt center',2x,'belt ocean fraction')
      do k = 1,nbelts,1
         write(modelout,787) midlat(k),focean(k)
 787     format(11x,f5.1,19x,f5.3)
      enddo
	  
	  
	  END ! ends subroutine