       PROGRAM TOTAL_EBM
! by Ramses Ramirez  ..  WHat is the BIOTIC drawdown of CO2???? Say,  from plants?
!---------------------------------------------------------------------------------------
implicit none

integer, parameter :: nbelts = 18 ! 18 latitudinal belts

integer ::  i,j,k, n, igeog, snowballflag, seasonflag, &
oceandata, fresneldata, modelout, seasonout, co2cloudsout, &
inputebm, last, n1, n2, nfile, nwthr, nedge, &
  imco2, nh, orbits, orbitcount

!integer :: wpred


real , parameter :: pi = 3.14159
real , parameter :: grav = 6.6732e-11 ! gravitational constant (mks units)
real ,  parameter ::  msun = 1.9891e30 !solar mass (kg)
real,  parameter :: q0 = 1360. ! solar flux on Earth at 1 AU (W/m^2)
real, parameter :: cnvg = 1.e-1
real, parameter :: mp = 1.67e-24 !mass of proton
real, parameter :: twopi = 2*pi
      

	  
real :: latrad,  lat, midlatrad,  midlat, x, coastlat, ocean,focean, &
delx, area, h20alb, temp, c, tprime, tprimeave, t2prime, alpha, &
beta, cw, cl, v, wco2, avemol0, avemol, hcp, hcp0, hcpco2,zrad, tcalc, &
trueanom,r, thetawin,  dec, decangle, tempsum,  fluxsum,  albsum, &
globwthrate, co2cldsum, tstrat, tlapse, htemp,hpco2, psl, patm, &
psat, as, q, decold, decmid, decnew,  fluxave,  co2cldave, &
albave, tempave,  fluxavesum, albavesum, tempavesum,  wthratesum, &
 co2cldavesum, ann_tempave, ann_albave,  ann_fluxave,  ann_irave, &
 ann_wthrateave,  ann_co2cldave, wthr, rot0, shtempave, nhtempave, &
 prevtempave, bb, sumsteps, numsteps,cose, tempi, po, nstep,P, lday


real :: mu, ir, s,atoa, surfalb, acloud, zntempmin, zntempmax, &
zntempsum, zntempave,  zndecmax, zndecmin, obstemp, iceline, &
fice, wthrate, warea, phi, h, z, oceanalb, snowalb, ci 


real :: landalb, icealb, irsum, iravesum, irave, icelat, &
icesht,asum, d0, d, a, a0, w, relsolcon, nwrite, twrite, &
dt, t, ecc, tperi, e, m, peri, rot, tend, obl, cloudir, pco2,& 
groundalb,solcon, landsnowfrac, fcloud, oblrad, tlast, &
fwthr


dimension :: latrad(0:nbelts + 1), lat(0:nbelts + 1), midlatrad(nbelts), &  
       midlat(nbelts), x(0:nbelts+1), focean(nbelts), delx(0:nbelts), &
	   h20alb(0:90), area(nbelts), temp(0:nbelts+1),  c(nbelts),  &
	   tprime(0:nbelts), tprimeave(nbelts), t2prime(nbelts),  &
	   ir(nbelts), mu(nbelts), s(nbelts), atoa(nbelts), surfalb(nbelts), &
       acloud(nbelts), zntempmin(nbelts), zntempmax(nbelts), &
	   zntempsum(nbelts), zntempave(0:nbelts+1), zndecmax(nbelts), &
	   zndecmin(nbelts), obstemp(nbelts), iceline(0:5), fice(nbelts), &
	   wthrate(nbelts), warea(nbelts), imco2(nbelts)
	   
	   ! Arrays with nbelts +2 include 90 degree(nbelts+2) and -90  degree (1) end points(1 and nbelts+2).
	   
 character(len=18)::aa
 character  header*80

 
! INPUT FILE FLAGS----------------------------------------
inputebm = 1
oceandata = 2
fresneldata = 3

! OUTPUT FILE FLAGS
modelout = 4
seasonout = 5
 co2cloudsout = 6
!----------------------------------------------------------

! READING INPUT FILE INITIAL PARAMETERS AND FLAGS
open(inputebm, file = 'input_ebm.dat', status = 'old')

do i = 1,2
read(inputebm,*) ! skips 2 lines before reading lines
enddo

read(inputebm,*)aa, seasonflag
read(inputebm,*)aa, snowballflag
read(inputebm,*)aa, igeog
read(inputebm,*)aa, pco2
read(inputebm,*)aa, tempi
read(inputebm,*)aa, lday
read(inputebm,*)aa, a0
read(inputebm,*)aa, ecc
read(inputebm,*)aa, peri
read(inputebm,*)aa, obl
read(inputebm,*)aa, ocean
read(inputebm,*)aa, groundalb
read(inputebm,*)aa, landsnowfrac
read(inputebm,*)aa, fcloud
read(inputebm,*)aa, solcon
read(inputebm,*)aa, numsteps
read(inputebm,*)aa, orbits
read(inputebm,*)aa, cloudir

!---------------------------------------------------------------------	 
! initialize variables
d0 = 0.58 !  thermal diffusion coefficient
a = a0*1.49597892E11 ! in meters
alpha = -0.078 !alpha: cloud albedo = alpha + beta*zrad....WHAT THE HECK IS THIS??? WHAT Is ALPHA, BETA, and ZRAD? See papers.. this explained there
beta = 0.65
 cw = 2.1e8       !heat capacity over ocean
 cl = 5.25e6      !heat capacity over land
v =   3.3e14  !volcanic outgassing (g/year) [from Holland (1978)]
wco2 = 9.49e14  !carbonate-silicate weathering constant
avemol0 = 28.965*mp  !mean-molecular weight of present atmosphere (g)
hcp0 = 0.2401    !heat capacity of present atmosphere (cal/g K)
hcpco2 = 0.2105  !heat capacity of co2 (cal/g K)
last = 0       ! At last step of do while loop, this turns to "1"
prevtempave = 0. ! initial previous temperature.
zrad = 0.
avemol =0.
imco2(1:nbelts) = 0
psat = 0.
snowalb = 0.663  ! ..dirty snow.. changed this from 0.7 as per Caldiera & Kasting (JDH)
icealb = snowalb  
ann_tempave =  200. ! just some initial  value for annual temperature  average that  is greater than  zero.
 po = 1. ! total surface  pressure for the Earth (1 bar)
 rot0 = 7.27e-5 ! rotation  rate for Earth (rads/sec)
zntempmax(1:nbelts) = 0. ! initial guess on  max zonal temperature 
zntempmin(1:nbelts) = 500. ! initial guess on minimum  zonal temperature. An arbitarily high value.
zntempsum(1:nbelts) = 0. ! initial guess on zonal  temperature sum
albavesum = 0.  ! average albedo sum term initial value
fluxavesum = 0. ! average flux sum term initial  value
iravesum = 0.   ! average outgoing infrared flux initial value
co2cldavesum = 0. ! CO2 clouds average sum initial value
wthratesum = 0. ! weathering rate sum initial value
decold = 0.0  ! old declination value
decmid = 0.0  ! mid declination value
decnew = 0.0  ! new declination value
nfile = 0
nwthr = 0     ! seasonal/orbital counter
nstep = 0     ! time step  counter within an orbit.  Reset at the beginning of each new orbit.
!wpred = 0    ! if hasn't read WRAPUP yet

! INPUT DATA
! OPEN OCEAN DATA TABLE---------------------------------


open(oceandata, file = 'data/oceans.dat',status= 'old')



!---------------------------------------------------------
      ! Starting temperatures
      oblrad = obl*pi/180.
      if (snowballflag.eq.1) then
	     temp(0:nbelts+1) = 233.
      else
             temp(0:nbelts+1) = tempi  ! start at some beginning temperature if not snowball 
      end if




open(modelout, file ='out/model.out', status='unknown')
open(seasonout, file ='out/seasons.out', status= 'unknown')
open(co2cloudsout, file ='out/co2clouds.out', status = 'unknown')

!  WRITE OBLIQUITY TO OUTPUT
      write(modelout,2) 'OBLIQUITY: ', obl, ' degrees'
 2    format(/ a,f5.2,a)

!  WRITE CO2-CLOUD FILE HEADER
      write(co2cloudsout,3)
 3    format(58x,'IMCO2')
      write(co2cloudsout,4)
 4    format(2x,'solar dec.(deg)',2x,'-85',1x,'-75',1x,'-65', &
     1x,'-55',1x,'-45',1x,'-35',1x,'-25',1x,'-15',2x,'-5',3x,'5',2x, &
     '15',2x,'25',2x,'35',2x,'45',2x,'55',2x,'65',2x,'75',2x,'85', &
     2x,'global cloud coverage' /)

	 
	 


	 
! READ FRESNEL REFLECTANCE TABLE------------------------
open(fresneldata, file = 'data/fresnel_reflct.dat', status='old')


do i = 1,2
read(fresneldata,*)  ! skips two lines in file
enddo 


      do i = 1,9
      n1 = 10*(i - 1)
      n2 = n1 + 9
      read(fresneldata,*)bb, (h20alb(j), j = n1,n2)
	  h20alb(n1:n2) =  h20alb(n1:n2)/90. !converting to ocean albedo..williams ebm divided by 100..I think that's wrong
      enddo
      h20alb(90) = 1.00	 !ocean albedo at 90 degrees incidence
	 
!      do  i = 1,90,1
!        write(60,*)h20alb(i)
!      enddo


!--------SET UP LATITUDINAL GRID (BELT BOUNDARIES)-----
!NEED ZERO INDEX FOR TPRIME and  TPRIME2 that uses  k-1 index

latrad(0) = -pi/2  !  -180 degrees(in radians)
lat(0) = latrad(0)*180./pi  ! -180 degrees


latrad(nbelts + 1) =  pi/2  !  -180 degrees(in radians)
lat(nbelts+1) = latrad(nbelts +1)*180./pi  ! -180 degrees

!  print *, lat(1), lat(nbelts+1)
  
! Creating 18 latitudinal bands 
do k = 1, nbelts, 1
  latrad(k) = latrad(0) + k*pi/nbelts     ! latitudinal (in rads)
  lat(k) = latrad(k)*180./pi ! latitudinal (in degrees)
  midlatrad(k) = latrad(0) + (1 + 2*(k-1))*pi/(2*nbelts)  !  center of latitudinal bands (in rads)
  midlat(k) = midlatrad(k)*180./pi !  center of latitudinal bands (in degrees)
enddo

! Fixing ..getting the right angles here for the beginning.
midlatrad(1) = -85*pi/180.
midlat(1) = -85 

!do j = 1, 18
!  write(40,*) midlat(j), midlatrad(j)
!enddo  

!do j = 0,18
!  write(40,*) lat(j), latrad(j)
!enddo  
 
 
! Latitudinal boundaries in sinusoidal space
x(0) = sin(latrad(0))
x(nbelts + 1) = sin(latrad(nbelts+1))
!print *, x(1), x(nbelts+1)


   
!write(40,*)x(0), x(nbelts+1)


!----------------------------------------------------
! SET UP GEOGRAPHY
      call geog(igeog, midlatrad, midlat, oceandata, coastlat, focean, latrad, lat, ocean, nbelts)


	  
!----------SET UP TEMPERATURE GRID-------------------


! redefine latitude grid to belt centers
      do k = 1, nbelts
	  latrad(k) = midlatrad(k)
	  lat(k) = midlat(k)
	  x(k) = sin(latrad(k))
!	  write(40,*)latrad(k), lat(k),x(k)
      enddo
	  

! calculate grid spacing
      do k = 0,nbelts
	  delx(k) = abs(x(k+1) - x(k))
!	  write(40,*)delx(k), x(k)    !why is delx offset from x?
      enddo
	  
! write grid data to output file
        write(modelout,140)	  
140     format(/ / '**LATITUDE GRID DATA**')
        write(modelout,142)
142    format(10x,'x',6x,'dx',4x,'latitude(rad)',2x,'latitude(deg)')
       do k = 0,nbelts+1
       write(modelout,143) k,x(k), delx(k),latrad(k),lat(k)
143	   format(3x,i2,2x,f6.3,2x,f6.3,5x,f6.3,8x,f5.1)
       enddo

	   
	  
!---------------------------------------------------------

! CALCULATE BELT AREAS - NORMALIZED TO PLANET SURFACE AREAS

       asum = 0.
       do k = 1,nbelts
       area(k) = abs(sin(latrad(k)+pi/(2*nbelts))- &
	   sin(latrad(k)-pi/(2*nbelts)))/2.
	   asum = area(k) + asum  ! summing all the areas.. adds up to 1!
!	   print *, area(k)
       enddo
!	   write(40,*)asum
!--------------------------------------------------------
! BEGIN INTEGRATION  AT VERNAL EQUINOX. FIRST CALCULATE TIME SINCE PERIHELION.

d = d0 ! initialize diffusion coefficient

rot = (360./lday)*(pi/180.) ! rotation rate  (rad/secs)

w = ((grav*msun)**0.5)/(a**1.5)  ! 2*pi/(orbital period(sec))
!print *, 2*pi/w  ! orbital period seconds

P = 2*pi/w  ! period for one orbit (seconds)
tend = orbits*P  ! total calculation time (seconds)

!numsteps = tend/5.

dt = tend/numsteps  ! the time step for each step

!write(60,*)orbits, tend, numsteps, dt

! write to co2clouds.out, at most, 1000 times
nwrite = amin0(1000, int((P)/dt))
twrite = (P)/nwrite

!write(50,*)twrite, nwrite, pi, w, dt


! Do integration for mean-annual case with variable obliquity
! taken  from Ward. Integration  is done in  the insolation
! section and uses a function defined  at the bottom of this code 
! obl = 0.
! dec = 0.

!print *, s

!write(40,*) dt, tend, numsteps

	  sumsteps = 0  ! counts the number of TOTAL steps. Unlike nstep, which counts all the steps  in ONE orbit
	  t = 0.  ! Time per each orbit (sec). Resets to 0 when one full orbit is achieved.
	  tcalc = 0.  ! Total elapsed time(sec)
          tlast = 0.  ! total time until twrite is achieved. However, in last orbit, goes to zero (sec)

	  orbitcount = 1 ! starting the very first orbit

!============BIG DO WHILE LOOP!!!!!!===================================================================================
	
    DO WHILE((tcalc.le.tend))
	
	
!	if (wpred.eq.1)then
!	last = 1 ! After  wrapup has been read, this last orbit is for the determination of zonal statistics
!	endif 
	

!!!! NO.. MUST CONVERGE FIRST TO PRINT ANNUAL STATISTICS BEFORE SETTING THIS FLAG!!!!! Last =1 is for ZONAL STATISTICS only
!	if (abs(prevtempave-ann_tempave).lt.cnvg)then
!	last = 1 ! When convergence is reached, set last flag to one.
!	tlast =  0.
!	endif
	
	sumsteps = sumsteps + 1

!	write(60,*)tcalc, t, last

!	t = t + dt      ! time stepping it through whether seasonal calculation or not...
!	tcalc = tcalc + dt
!	tlast = tlast +  dt  ! tlast is time in seconds

!	if(sumsteps.eq. numsteps)then
!	last = 1  !  PROBABLY SET TO 1 ALSO AFTER CONVERGENCE HAS  BEEN  ACHIEVED.. UPDATE THIS!!!
!	endif
	
	
!	write(40,*)last,t, tcalc, tlast, P, sumsteps, numsteps  ! T EXCEEDS P at step 19,998, 2 steps before last .eq. 1 !!!!
	! 3.15e7 sec in an Earth year.. I think it averages the stuff after every year..!!!
	
!------------------------------------------------------------------	!COME BACK.. MAY PREFER TO HAVE THIS LOGIC HERE INSTEAD OF IN WRAPUP
	
	
      if (seasonflag.eq.1)then    ! DO WHILE that steps in t.. do while t < tend , t = t + dt.. etc.. if seasons starts with 240 logic if not, starts with mean annua..then go into big loop


      cose = (cos(peri*pi/180.)+ecc)/(1.+ecc*cos(peri*pi/180.))  
	  e = acos(cose) ! eccentric anomaly (in radians)
      m = e - ecc*sin(e)   !**Kepler's Equation (m is the mean anomaly)
      tperi = -m/w    !**tperi is the time of perihelion (vern.eqnx. is 0)in seconds (w = 2*pi*f =  2*pi/Period)
	  !(tperi = -m*Period/2*pi... from tperi/P = m/(2*pi).. except here it is negative)
!	   write(60,*)t, e, m, ecc, tperi, w, peri, pi, ecc

! write(modelout, 232)
232 format(/ 'ORBITAL DATA')

! write(modelout, 233)
233  format(2x,'time (sec)',2x,'true anomaly (deg)',2x, &
     'solar declination (deg)',2x,'orbital distance (cm)',2x, &
       'irradiance (Wm^-2)')
	 
      if (tcalc.lt.tend) then
         t = t + dt
         tcalc = tcalc + dt
         tlast = tlast + dt
      end if
	  
	  
	       m = w*(t-tperi)  ! mean anomaly, fraction of orbit from periapsis
           if (m.gt.2*pi)then 
		   m = m - 2*pi   ! keeps mean anomaly between 0 and 2*pi radians
		   endif
	       
!		   write(60,*)m,ecc,e
	       call keplereqn(m,ecc,e)  ! output eccentric anomaly
           trueanom = acos((cos(e) - ecc)/(1 - ecc*cos(e)))
!	       write(70,*)m,ecc,e, trueanom 
			
!  correct trueanomaly for pi < m < 2pi
         if (m .gt. pi)then 
		 trueanom = 2*pi-trueanom
         endif
		 

            r = a*(1-ecc*cos(e)) ! orbital radius (in m)
            q = q0 * solcon * (a/r)**2  !solar flux on point in orbit
!			write(60,*)q
      !q = q0 * solcon * (a0/r)**2
            thetawin = peri*pi/180 + 3*pi/2
            dec = asin(-sin(obl*pi/180.)*cos(trueanom-thetawin))
            decangle = dec*180./pi
			
!            write(60,*)trueanom, decangle
			

			
			
!			write(60,*)t, trueanom*180./pi, decangle, r, q
255  format(2x,e9.3,6x,f7.3,14x,f7.3,16x,e11.6,13x,f9.3)

! At the beginning of each time step in the seasonal calculation, these sums are set to zero. Looking for sums of all LATITUDES within a time step.
            tempsum = 0. 
	        fluxsum = 0.
	        albsum = 0.
	        irsum = 0.
	        globwthrate = 0.
	        co2cldsum = 0.



         else ! if no seasons
	  
	  ! MEAN ANNUAL CALCULATION with variabe obliquity from Ward. Integration is done in the insolation section and uses a function defined at the bottom of this code.
            q = q0*solcon

      if (tcalc.lt.tend)then
      t = t + dt
      tcalc = tcalc + dt
      tlast = tlast +  dt
	  
      else
         write(*,*) 'Calculation time has elapsed.'
         CALL WRAPUP(last,k,ann_tempave,ann_albave,ann_fluxave,ann_irave,ann_wthrateave, &
	  wco2,d,pco2,nbelts,modelout,zntempmin, tcalc) ! only in  wrapup can last be made equal to 1 to loop over code one last time
      endif
	  
	  
      endif  ! ends season/no season if logic



!--- FINITE DIFFERENCING - ATMOSPHERIC and OCEANIC ADVECTIVE HEATING 

      do  k= 0,nbelts   !**first derivatives between grid points
         tprime(k) = (temp(k+1) - temp(k))/delx(k)
!		 if(sumsteps.eq.1)write(60,*) tprime(k), temp(k),  temp(k+1), delx(k)
      enddo
	  
!----------------------------------------------------------------------c
!  **THE BIG LATITUDE LOOP WITHIN EACH TIME STEP **.. LATER TRY TO  REVIEW THE  FINITE DIFFERENCING FOR 1st and 2nd derivatives
!----------------------------------------------------------------------c


      DO  k=1,nbelts,1   !**start belt loop 
                           !**first derivatives at grid points
         tprimeave(k) = (tprime(k)*delx(k) + tprime(k-1)* &
           delx(k-1))/(delx(k) + delx(k-1))
                           !**second derivatives at grid points
         t2prime(k) = ((((delx(k)/2)**2)-((delx(k-1)/2)**2))*(1-x(k)**2)* &
         tprimeave(k) + ((delx(k-1)/2)**2)*(1-(x(k)+(delx(k)/2))**2)*  &
         tprime(k) - (((delx(k)/2)**2)*(1-(x(k)-(delx(k-1)/2))**2))*   &
         tprime(k-1))/((delx(k)/2)*(delx(k-1)/2)*(delx(k)/2 + delx(k-1)/2)) 
		 
!		 if(sumsteps.eq.1)write(60,*)t2prime(k),tprimeave(k), delx(k-1)
!write(60,*)t2prime(k),tprimeave(k), delx(k-1), sumsteps

!----------------------------------------------------------------------c
!  OUTGOING INFRARED (Obtained from fits to Jim Kasting's radiative
!    -convective model earthtem_ir. The fit can be applied for
!    190 < T < 370 K and 10^-5 < PCO2 < 10 bars with an rms error
!    of 4.59 Wm^-2.)                                 
!
!      if ( temp(k) .le. 190.0 ) then
!        print *, k, ": ", temp(k), " Temp too low"
!      else if ( temp(k) .ge. 370.0 ) then
!        print *, k, ": ", temp(k), " Temp too high"
!      end if

       phi = log(pco2/3.3e-4)
	   
!	   write(40,*) pco2, phi
	   
	   
       ir(k) = -3.102011e-2 - 7.714727e-5*phi - &
       3.547406e-4*phi**2 - 3.442973e-3*phi**3 - &
       3.367567e-2*phi**4 - 2.794778*temp(k) - &
       3.244753e-3*phi*temp(k) + 2.229142e-3*phi**2*temp(k) + &
       9.173169e-3*phi**3*temp(k) - 1.631909e-4*phi**4*temp(k) + & 
       2.212108e-2*temp(k)**2 + 3.088497e-5*phi*temp(k)**2 - &
       2.789815e-5*phi**2*temp(k)**2 - 7.775195e-5*phi**3*temp(k)**2 + & 
       3.663871e-6*phi**4*temp(k)**2 - 3.361939e-5*temp(k)**3 - &
       1.679112e-7*phi*temp(k)**3 + 6.590999e-8*phi**2*temp(k)**3 + & 
       1.528125e-7*phi**3*temp(k)**3 - 9.255646e-9*phi**4*temp(k)**3 

       ir(k) = ir(k) - cloudir   !**reduction of outgoing-infrared by clouds (For 288 K on Earth, should get outgoing IR ~236 - 237 W/m^2)
	   
	   
!	   if(sumsteps.eq.1)write(60,*) temp(k),ir(k), cloudir, phi,  pco2
!
!----------------------------------------------------------------------c
!  DIURNALLY-AVERAGED ZENITH ANGLE                            

      if ((tan(dec)*tan(asin(x(k)))).ge.1.) then 
         h = pi
         mu(k) = x(k)*sin(dec) + cos(asin(x(k)))*cos(dec)*sin(h)/h
      else if ((tan(dec)*tan(asin(x(k)))).le.-1.) then 
         h = 0.
         mu(k) = 0.
      else
         h = acos(-x(k)*tan(dec)*((1-x(k)**2)**(-0.5)))
         mu(k) = x(k)*sin(dec) + cos(asin(x(k)))*cos(dec)*sin(h)/h
      end if
!  
      z = acos(mu(k))*180./pi  ! solar zenith angle (degrees)
      zrad = acos(mu(k)) ! solar zenith angle (rads)
	  
!	  if(sumsteps.eq.1)write(60,*)zrad, z,  acos(mu(k)), h
                 
!----------------------------------------------------------------------c
!  HEAT CAPACITY and SURFACE ALBEDO                    
!
      oceanalb = h20alb(int(z))  ! ocean albedo (in degrees, reflectance)
!if(sumsteps.eq.1)write(60,*)z,int(z), oceanalb

! DETERMINE ICE FRACTION AS A FUNCTION OF  TEMPERATURE=====================================================

      ! Added this check so fractional ice cover would only 
      ! occur for 263 < T < 273.  (JDH)
!      if(focean(k).ge.0.5) then        
!         if (temp(k).ge.238.15) then
!            fice(k) = 0
!         else if (temp(k).le.238.15) then
!            fice(k) = 1
!         else
!            !fice(k) = 1. - exp((temp(k)-273.15)/10.)
!            fice(k) = 0.1*(248.15 - temp(k)) ! uncomment for a linear decrease
!         end if 
!      else
         if (temp(k).ge.273.15) then
            fice(k) = 0
         else if (temp(k).le.263.15) then
            fice(k) = 1
         else
            fice(k) = 1. - exp((temp(k)-273.15)/10.)
         end if
!      end if   
!====================================================================
 
!if(sumsteps.eq.1)write(60,*)temp(k),  fice(k) 

      acloud(k) = alpha + beta*zrad
	  
      if (temp(k).le.263.15) then
!      if (temp(k).le.273.15) then	  
	  ! LAND WITH STABLE SNOW COVER; SEA-ICE WITH SNOW COVER
      landalb = snowalb*landsnowfrac + groundalb*(1 - landsnowfrac)
      icealb = snowalb

      ci = 1.05e7  ! thermal heat capacity of ice  (cl and  cw is the same for  land and water,  respectively)
      c(k) = (1-fice(k))*focean(k)*cw + fice(k)*focean(k)*ci + &
       (1 - focean(k))*cl
      fwthr = 0.  ! fraction of surface for weathering
	  
	  surfalb(k) = max(((1-focean(k))*landalb + &
      focean(k)*((1-fice(k))*oceanalb + fice(k)*icealb)), &
      fcloud*acloud(k))
	 
	  
      elseif(temp(k).ge.273.15)then ! if temperature are above273.15 K then
      landalb = groundalb
      fice(k) = 0.
      fwthr = 1.       !**100% if T > 273K, 0% otherwise
      c(k) = focean(k)*cw + (1 - focean(k))*cl
	  
	  surfalb(k) = (1-fcloud)*((1-focean(k))*landalb + &  
      focean(k)*((1-fice(k))*oceanalb + fice(k)*icealb)) + &
      fcloud*acloud(k)     
	  
	  elseif((temp(k).gt.263.15).and.(temp(k).lt.273.15))then
	  landalb = groundalb
      fice(k) = 0.
      fwthr = 1. 
	  c(k) = focean(k)*cw + (1 - focean(k))*cl
	  
	  
	  surfalb(k) = max(((1-focean(k))*landalb + &
      focean(k)*((1-fice(k))*oceanalb + fice(k)*icealb)), &
      fcloud*acloud(k))
      endif 


!=======================================================================     
      


!if(sumsteps.eq.1)write(60,*)focean(k),oceanalb,icealb,fice(k),fcloud      

      if(last.ne.1)then 
      CALL WEATHERING(area, k, nbelts, focean, fwthr, wthrate, warea, temp)
      endif


!----------------------------------------------------------------------c
!  CO2 CLOUDS, IF LAST LOOP (Determines if atmospheric conditions allow for CO2 cloud formation and says where they are).. Be good to eventually put in equivalent for water clouds, right?
      if(last.eq.1)then    
	  
      tstrat = -188.0968076496237 - 1.954500243043531*phi + &
       3.809523755282467*phi**2 + 2.327695124784982*temp(k) + & 
       0.0003732556589321478*phi*temp(k) - & 
       0.02855747090737606*phi**2*temp(k) - & 
       0.00332927322412119*temp(k)**2 + & 
       0.00002213983847712599*phi*temp(k)**2 + & 
       0.0000460529090383333*phi**2*temp(k)**2
	   
!	   write(60,*)tstrat,  phi,  temp(k), lat(k)
	   
!	   write(40,*)'WHAT THE HECK?'  !ok.. so last.eq.1 working here.
	   
!
      nh = 0
      tlapse = 6.5    !(lapse rate)degrees Kelvin per kilometer.. BUT THIS SHOULD DEPEND on THE PLANET, RIGHT? 6.5K/km is  average for Earth!!!
	  htemp = temp(k) - tlapse*nh
	  
!	  print *, htemp, temp(k), k, nh
!	  pause
	  

	  
	  if (htemp.lt.tstrat)then   ! which skips the entire do while loop condition  BELOW because CO2 clouds do not form with this condition
	  
      CALL WEATHERING(area, k, nbelts, focean, fwthr, wthrate, warea, temp)

      endif ! ends htemp/tstrat if logic
	  
!	  	  write(60,*) avemol, htemp,tstrat,nh
!=============================================================================	  
	  do while(htemp.ge.tstrat)

	  

      hpco2 = pco2*((temp(k)-tlapse*nh)/temp(k))** &
       (avemol*981./(1.38e-16*tlapse*1.e-5))
	   
	   


	        ! SUBROUTINE SATCO2 from Jim Kasting "earthtem_0.f"
			
	        !   VAPOR PRESSURE OVER LIQUID
	        if (htemp.gt.216.56)then 
            psl = 3.128082 - 867.2124/htemp + 1.865612e-2*htemp - &
            7.248820e-5*htemp**2 + 9.3e-8*htemp**3

            !  VAPOR PRESSURE OVER SOLID 
            elseif (htemp.lt.216.56)then  

           ! (Altered to match vapor pressure over liquid at triple point)
            psl = 6.760956-1284.07/(htemp-4.718) + 1.256e-4*(htemp-143.15)

            endif 
			
         patm = 10.**psl
         psat = 1.013*patm
			
        imco2(k) = 0
		
		
        if (hpco2.ge.psat) then      !**co2 clouds form
        imco2(k) = 1  ! this labels which latitude CO2 clouds form
        CALL WEATHERING(area, k, nbelts, focean, fwthr, wthrate, warea, temp)

        exit ! THIS do while loop is  exited 
		
        else ! if hpco2 lt psat then clouds do not form and need to go to a higher (cooler) altitude (1 km higher) and check again
		nh = nh + 1
	    htemp = temp(k) - tlapse*nh
!		     write(60,*) hpco2,psat,htemp,tstrat,nh,lat(k),imco2(k)
	      if (htemp.lt.tstrat)then
	      CALL WEATHERING(area, k, nbelts, focean, fwthr, wthrate, warea, temp)
	      exit  ! THIS do while loop is  exited 
          endif 
		  
        endif  ! end if condition for CO2 cloud formation
		
		

       enddo ! ends do  while of htemp and tstrat loop!!
	  
!=============================================================================	  
	  
	   endif ! ends last logic for CO2 clouds

!----------------------------------------------------------------------c
!  TOP-OF-ATMOSPHERE ALBEDO (Obtained from fits to Jim Kasting's radiative
!    -convective model 'earthtem_alb.' Both fits can be applied for
!    10^-5 < pco2 < 10 bars, 0 < surfalb < 1, and 0 < z < 90 degrees
!    with r.m.s. errors (given in planetary-average incident solar
!    flux {340 W/m^-2}) of 7.58 and 4.66 Watts/m^2 for the low and
!    high temperature regimes respectively.

      as = surfalb(k)
      ! Added this check for out of range temperatures (JDH)
!      if(temp(k).ge.370) then
!         print *, 'Temp too high!:',k, temp(k)
!         goto 530
!      end if
      if(temp(k).ge.280.) then  !**goto high-temp fit
	  
	  
	   atoa(k) = 1.108210 + 1.517222*as + 7.588651e-2*as**2 - &
        1.867039e-1*mu(k)+2.098557e-1*as*mu(k)+6.329810e-2*mu(k)**2 + &  
        1.970523e-2*pco2 - 3.135482e-2*as*pco2 - &
        1.021418e-2*mu(k)*pco2 - 4.132671e-4*pco2**2 - & 
        5.79932e-3*temp(k) - 3.709826e-3*as*temp(k) - &
        1.133523e-4*mu(k)*temp(k) + 5.371405e-5*pco2*temp(k) + &
        9.269027e-6*temp(k)**2
	 
      else ! for low temp-fit (lt <  280 K) 

      atoa(k) = -6.891041e-1 + 1.046004*as + 7.323946e-2*as**2 - &
        2.889937e-1*mu(k)+2.012211e-1*as*mu(k)+8.121773e-2*mu(k)**2 - &  
        2.837280e-3*pco2 - 3.741199e-2*as*pco2 - &
        6.349855e-3*mu(k)*pco2 + 6.581704e-4*pco2**2 + & 
        7.805398e-3*temp(k) - 1.850840e-3*as*temp(k) + &
        1.364872e-4*mu(k)*temp(k) + 9.858050e-5*pco2*temp(k) - & 
        1.655457e-5*temp(k)**2
      endif
	  
!if(sumsteps.eq.1)write(60,*)temp(k), atoa(k),as,pco2   

!----------------------------------------------------------------------c
!  DIURNALLY-AVERAGED INSOLATION 
      
      if (seasonflag.eq.1) then
         s(k) = (q/pi)*(x(k)*sin(dec)*h + &   !essentially eqn a8 in williams/kasting
             cos(asin(x(k)))*cos(dec)*sin(h))
		  
		 
!           if(sumsteps.eq.1)write(60,*)s(k), x(k),dec,h		  
		  	 
			 
      else ! if no seasons 
         ! for mean annual do the Ward isolation integration (JDH)
         call qtrap(sumsteps,0, twopi, s(k), latrad(k), oblrad)
         s(k) = (q/(2*(pi**2))) * 1/(sqrt(1 - ecc**2)) * s(k)
         !print *, 'insolation at', lat(k), ':', s(k)
      end if
!
!----------------------------------------------------------------------c
!  SURFACE TEMPERATURE - SOLVE ENERGY-BALANCE EQUATION 

      temp(k) = (d*t2prime(k)-ir(k)+s(k)*(1-atoa(k)))*dt/c(k) + temp(k)  !main energy balance equation

!      if(last.eq.1)write(40,*)zntempmin(1:nbelts),k,last
	  
!if(sumsteps.eq.1)write(60,*)temp(k),  ir(k), s(k), atoa(k), c(k), d, t2prime(k)     
	  

!  SUM FOR GLOBAL AVERAGING  (WITHIN A GIVEN TIME STEP.. SUMMING AREA CONTRIBUTION OVER ALL LATITUDES TO GET GLOBAL IR, FLUX, PALB, TEMP..etc)
      irsum = irsum + area(k)*ir(k)
      fluxsum = fluxsum + area(k)*s(k)
      albsum = albsum + area(k)*s(k)*atoa(k)  ! albsum is flux upward, reflected to space (s(k)*atoa(k) scaled by area
      tempsum = tempsum + area(k)*temp(k)
      globwthrate = globwthrate + wthrate(k)
      co2cldsum = co2cldsum + area(k)*imco2(k)*s(k)  
	  
         
!	  if(sumsteps.eq.1)write(60,*)irsum, tempsum, fluxsum, k
           

      if(last.eq.1)then !  ZONAL STATISTICS - if last ORBIT====================================
 

      zntempmin(k) = amin1(zntempmin(k),temp(k))
	  
      if (zntempmin(k).eq.temp(k))then 
	  zndecmin(k) = decangle
      zntempmax(k) = amax1(zntempmax(k),temp(k))
      endif
	  
      if (zntempmax(k).eq.temp(k))then 
      zndecmax(k) = decangle
      zntempsum(k) = zntempsum(k) + temp(k)
      endif  
	  ! With this logic, zntempmin and zntempmax BOTH start out the same
!	      write(60,*)zntempmin(k), zntempmax(k), temp(k),decangle,zndecmin(k), zndecmax(k),k	  
	  
      endif  ! ends if logic for zonal statistics===========================================
	  

!	  write(60,*)  

       ENDDO  
!======**end of BIG belt loop=================================================  
	   
!---------------------------------------------------------	   

	   
!  **set pole temps equal to adjacent belt temps
      temp(0) = temp(1)   
      temp(nbelts+1) = temp(nbelts)

!  WRITE OUTPUT FILES - ONLY ON LAST LOOP      
!      if(.not.last ) goto 710
!      if(tlast.lt.twrite) goto 710   !!!!!!! UNDERSTAND WHAT THIS STATEMENT MEANS!!!!????????????


!----------------------------------------------------------------------------------	  
      if((last.eq.1).or.(tlast.ge.twrite))then  ! do this stuff if in the last ORBIT or when tlast <= twrite
!	  	   write(50,*)last, sumsteps, tlast, twrite	  

!	  write(50,*)tlast,twrite, sumsteps, last,dt   ! when last equals 1 it DOES come here...sumsteps still 2.e4.. BY THE TIME IT GETS HERE only ONE VALUE for SUMSTEPS because it is at LAST STEP BY DEFINITION
      tlast = 0.  ! resets tlast
!
!  EQUINOXES AND SOLSTICES
!
! added seasons check here (JDH)====== 
! AS FAR AS I CAN TELL, THIS SUBSECTION IS NOT EXECUTED. 
! No  values for  decold,  decmid, or decnew anywhere in code.
      if (seasonflag.eq.1) then
         decold = decmid
         decmid = decnew
         decnew = decangle

         if (((abs(decmid).gt.(abs(decold))).and. &
         (abs(decmid).gt.(abs(decnew)))).or.(decnew*decmid.le.0.))then 
     
!     write(15,610) file(nfile),decangle
 610        format('data written to file ',a8, ' at declination ',f6.2, &
                ' degrees')
!            print *, 'nfile=', nfile
!            open(unit=nfile+7,file=file(nfile),status='unknown')
!            do k = 1,nbelts,1
!               write(nfile+7,615) latrad(k),temp(k),atoa(k),ir(k),
!                    d*t2prime(k),s(k)*(1-atoa(k))
!            enddo
 615           format(f6.2,5(3x,f7.3))

            nfile = nfile + 1
			
!			write(60,*)nfile, decold, decmid, decnew, decangle,last
         end if         
      end if  
! end if logic seasons===============================================

!  CO2-CLOUD DATA
      write(co2cloudsout,630) decangle,imco2,co2cldsum/fluxsum
 630  format(3x,f6.2,9x,18(3x,i1),9x,f5.3)
 
!      write(60,*)decangle, co2cldsum, fluxsum  ! still 2.e4
	  
      endif !! ends last twrite logic SEASONS IF STUFF========================================
!----------------------------------------------------------------------c


!  GLOBAL AVERAGING (Done every time step)



     ! The global sum values over all latitudes for a given time step = average global values for a given time step 
      irave = irsum
      fluxave = fluxsum
      co2cldave = co2cldsum/fluxave
      albave = albsum/fluxave   ! flux up/fluwn is albedo average
      tempave = tempsum

      ! summing all of these global averages over ALL time steps within each orbit
      iravesum = iravesum + irave
      fluxavesum = fluxavesum + fluxave
      albavesum = albavesum + albave
      tempavesum = tempavesum + tempave
      wthratesum = wthratesum + globwthrate
      co2cldavesum = co2cldavesum + co2cldave
      nstep = nstep + 1     ! Go to next time step within orbit. Nstep goes back to ZERO after each revolution around star


!  if(sumsteps.eq.1)write(60,*)irave, albsum, fluxave, albave


 !zntempmin(5), zntempmax(5)   ! zntempmin, zntempmax showed up not at last step, last step here really is the entire last ORBIT!!!!!
	  
!	  write(60,*)irave,fluxave,co2cldave, albave,tempave, sumsteps
!	  write(60,*)iravesum,fluxavesum,albavesum,tempavesum,wthratesum,co2cldavesum,sumsteps

! DEFAULT	  
!	  write(50,*)sumsteps, t, P     !t exceeds P at step 19,998, 2 steps before last!!!.. Get bunch of SUMSTEP values again here cuz not in above last.eq.1 logic
!     write(50,*)irave,iravesum, wthratesum, nstep
	  
!====================================================================================================================================================================
!====================================================================================================================================================================

      if(t.lt.P)then    !**one orbit since last averaging.. do nothing more and go to the end to restart the loop, without initializing any sums...



		
!  ANNUAL AVERAGING (comes here after the end of an orbit=====================================================
      elseif (t.ge.P)then !**one orbit since last averaging.
      ann_tempave = tempavesum/nstep
      ann_albave = albavesum/nstep
      ann_fluxave = fluxavesum/nstep    ! annual averaging occurs after ONE full stellar revolution
      ann_irave = iravesum/nstep
      ann_wthrateave = wthratesum/nstep
      ann_co2cldave = co2cldavesum/nstep

!     write(60,*)ann_tempave,ann_albave, ann_fluxave, ann_irave 
!	 write(40,*)ann_wthrateave,ann_co2cldave, pco2,d, nstep
       
!      write(60,*)ann_tempave, tempavesum, nstep


!  ADJUST PCO2 EVERY 5 ORBITAL CYCLES.. WHY IS THIS LOGIC DONE????
!---------------------------------------	  	  
      if(nwthr.ge.5)then
      nwthr = 0 ! after 5th orbit, restart counter to  0
      wthr = wco2*ann_wthrateave
	  
      if (wthr.lt.v)then
      if(wthr.lt.1.e-2) then
	  wthr=1.

      endif  !if wthr lt1.e-2
      endif ! if wthr lt.v
      endif ! if nwthr ge. 5

      if(nwthr.lt.5)then
	  nwthr = nwthr + 1   ! counter goes up every season  (orbit)
      endif ! END SEASONAL CYCLE PCO2 IF LOGIC
!-------------------------------------------

!   ADJUST DIFFUSION COEFFICIENT
      avemol = mp*(28.965+44.*pco2)/(1.+pco2) ! molecular weight expression
      hcp = (hcp0 + pco2*hcpco2)/(1.+pco2)  ! heat capacity
!      write(*,*) pco2,d
      d = d0*((1.+ pco2)/po)*((avemol0/avemol)**2)*(hcp/hcp0)* &   ! Assumes p = 1bar + pcO2. Need to update for other planets
        (rot0/rot)**2

!      if(orbitcount.eq.2)write(60,*)avemol, hcp, hcp0, rot0, rot, d, d0, mp, pco2

!=============================================================	  



!      If at the end of the orbit (about to start next t.ge.P) and not converged, set prevtempave = ann_tempave. Otherwise, if converges, stop calculation early.. you are done!



       if(abs(prevtempave-ann_tempave).lt.cnvg)then  ! If annual temps converged, then write out annual statistics!!! So long it has not been read already


        CALL WRAPUP(last,k,ann_tempave,ann_albave,ann_fluxave,ann_irave,ann_wthrateave, &
	  wco2,d,pco2,nbelts,modelout,zntempmin, tcalc) ! COME HERE TO READ ANNUAL/ORBITAL STATISTICS.. THEN HAVE TO RE-LOOP TO COMPUTE THE ZONAL STATISTICS BY COMPLETING ONE MORE ORBIT!!!

         
       elseif (abs(prevtempave-ann_tempave).gt.cnvg)then  ! if annual  temps not converged,  keep going
               prevtempave = ann_tempave 


        endif

!      write(40,*)prevtempave, ann_tempave, tcalc    .. CODE DOES NOT WORK cause WPRED REMOVED.. NO LAST.eq.1 INITIATED


 
	  
       if(last.eq.1) then  ! The last step of the last orbit (t ge. P condition).. PROBABLY HASN'T CONVERGED IF IT GETS TO THIS POINT!!!!!!!!!!!!!!
!------------------------------------------------------------------		   
	    ! set up output files for last step
     	
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
	  
	  

!     zntempave(1:nbelts) = zntempsum(1:nbelts)/nstep   ! WHAT THE HECK?? This is why iceball is being wrong!!
  zntempave(1:nbelts) = 0.5*(zntempmin(1:nbelts) +  zntempmax(1:nbelts))	 
! 	     write(40,*)zntempave(1:18)	 
		 
      do  k = 1, nbelts, 1	 
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
!--------------------------------------------------------------------------------	  
	  
!	  if(abs(prevtempave-ann_tempave).lt.cnvg)then
      CALL WRAPUP(last,k,ann_tempave,ann_albave,ann_fluxave,ann_irave,ann_wthrateave, &
	  wco2,d,pco2,nbelts,modelout,zntempmin, tcalc) ! only in  wrapup can last be made equal to 1 to loop over code one last time
	  
!	  write(40,*)zntempmin(1:nbelts),last
!      else
!	  prevtempave = ann_tempave 
!      endif   ! end if for convergence check.

	  
	  if(last.eq.1)stop ! after convergence has been achieved
	  
!	  write(40,*)prevtempave, ann_tempave


     elseif (last.ne.1)then! and t.ge P... So end of orbit, but not last  orbit 
	 
      fluxavesum = 0.  ! all of these are reset before the next orbit (except co2cldavesum).
      albavesum = 0.
      tempavesum = 0.
      iravesum = 0.
      wthratesum = 0.
      t = 0
      nstep = 0

	  
      orbitcount = orbitcount + 1   ! go to next orbit
	  
      endif ! ends last.eq.1 logic 

      endif  ! ends t P if logic 
	  


	  
       END DO  ! ends BIG DO  WHILE LOOP
!-------------------------------------------------------------------------------------------

       END ! Ends program

!-----------------------------------------------------------------------------------------

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


!------------------------------------------------------------------------------------------------

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
 
!      wpred = 1 ! WRAPUP has been read. Now go through one more orbit to obtain zonal statistics.
	        end   ! ENDS SUBROUTINE
 
!-------------------------------------------------------------------------------------


  
       




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




!----------------------------------------------------------------------------------

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


!---------------------------------------------------------------


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
































