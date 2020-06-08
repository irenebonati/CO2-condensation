       PROGRAM TOTAL_EBM_constantpCO2_36belts_Irene
!=======FEW THINGS: 
! 1) reduce fracd0 to (0.12/0.58)~ 0.2 for when warm start annual mean Ts < 273 K
! 2) Use igeog = 4, water world ocean = 1.0	   
! 3) Use hybrid data files for both warm and cold starts!
! 4) removed extraneous icoldflag logic in the beginning and end.
! 5) Base fracd0 should be "1"
! 6) Put in alternative ice albedo feedback
! 7) OPTIONAL: Added better fracd0 annual add...
!===================
! by Ramses Ramirez  ..  WHat is the BIOTIC drawdown of CO2???? Say,  from plants? 
! How about check Sellers and other book for ocean coverage per lat?
! Look at weathering rate parameterizations
! Put topographic function.. to get the T differences due to elevation for the various latitudes
!---------------------------------------------------------------------------------------
implicit none

integer, parameter :: nbelts = 36 ! 36 latitudinal belts
integer ::  i,j,k, n, igeog, snowballflag, seasonflag, &
oceandata, fresneldata, modelout, seasonout, co2iceout, &
inputebm, last, n1, n2, nfile, nwthr, nedge, co2siceout, &
  imco2i, imco2l, imco2s, nh, orbits, orbitcount, nstep, fstep, &
co2lout, icoldflag


real , parameter :: pi = 3.141592653589793
real , parameter :: grav = 6.6732e-11 ! gravitational constant (mks units)
real ,  parameter ::  msun = 1.9891e30 !solar mass (kg)
real,  parameter :: q0 = 1360. ! solar flux on Earth at 1 AU (W/m^2)
real, parameter :: cnvg = 1.e-1  
real, parameter :: mp = 1.67e-24 !mass of proton
real, parameter :: twopi = 2*pi
real, parameter :: Rgas = 8.314 !J/mol/K
!real, parameter :: L = 43655 ! J/mol
real, parameter :: po_star = 1.4e11 !Pa
	  
real*4 :: latrad,  lat, midlatrad,  midlat, x, coastlat, ocean,focean, &
delx, area, h20alb, temp, c, tprime, tprimeave, t2prime, alpha, &
beta, cw, cl, v, wco2, avemol0, avemol, hcp, hcp0, hcpco2,zrad, tcalc, &
trueanom,r, thetawin,  dec, decangle, tempsum,  fluxsum,  albsum, &
globwthrate, co2icesum, tstrat, tlapse, htemp,hpco2, psl, patm, &
psat, as, q, decold, decmid, decnew,  fluxave,  co2iceave, &
albave, tempave,  fluxavesum, albavesum, tempavesum,  wthratesum, &
 co2iceavesum, ann_tempave, ann_albave,  ann_fluxave,  ann_irave, &
 ann_wthrateave,  ann_co2iceave, wthr, rot0, shtempave, nhtempave, &
 prevtempave, bb, sumsteps, numsteps,cose, tempi, po, P, lday, compcnvg, &
 conv, Pdry, b, pco2sum, wco2e, ann_pco2ave, pco2avesum, &
 pco2ave, fdge,  Ptot, g, pco2r, ph2o, fnc, fco2, hcph2o, ann_ph2oave, &
 ph2oavesum, ph2oave, ph2osum, psurf, ann_psurfave, psurfavesum, psurfave, &
 psurfsum, co2lave, co2lsum, co2lavesum, ann_co2lave, L, cpn2, cpco2, &
 fn2, cptot, gamd, co2sicesum, co2siceave, co2siceavesum, ann_co2siceave, subl, &
  icealbco2, liql, fracd0, sumd0  !c-rr added fracd0, sumd0, and liql 6/16/2019 
 

real*4 :: mu, ir, s,atoa, surfalb, acloud, zntempmin, zntempmax, &
zntempsum, zntempave,  zndecmax, zndecmin, obstemp, iceline, &
fice, wthrate, warea, phi, h, z, oceanalb, snowalb, ci, Lum,  &
 temp2,  zice,  delp, aice_vis, aice_nir, fvis, fir, zicesum, dtlong, &
tlong, dtau

real*4 :: landalb, icealb, irsum, iravesum, irave, icelat, &
icesht,asum, d0, d, a, a0, w, relsolcon, nwrite, twrite, &
dt, t, tperi, peri, rot, tend, obl, cloudir, pco2,& 
groundalb,solcon, landsnowfrac, fcloud, oblrad, tlast, &
fwthr, pco2i, smass, mstar, newice, CO2ps


real*4 :: K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, &
K12, K13, K14, K15, K16
real*4 :: N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8, N_9, N_10, N_11, &
N_12, N_13, N_14, N_15, N_16

real*4::  fracd0sum, fracd0ave, fracd0avesum, ann_fracd0ave

real*4:: sumnewice
	   
real*4:: olr,palb, aaa, bbb, ccc, ddd, eee, fff, ggg, hhh, Z0, Z1, Z2, & 
Z3, Z4, PLOGL, PLOGR, pco2rlog, NA, NB, NC, ND, DEN, zyy, tempstrat
integer:: OLRparam, TT, PP, kk, TL, PL, TR, PR, PALBparam, ALB, ZY
integer :: ZYL, ZYR, AL, AR, TSTRATparam
integer, parameter :: sizep = 16, sizet = 25, sizezy = 19, sizesalb = 21
dimension:: olr(sizet,sizep)
dimension:: palb(sizet,sizep, sizezy,sizesalb)
dimension:: tempstrat(sizet,sizep)
dimension:: fracd0(nbelts)

real ::  pressgrid(sizep)
real :: tempgrid(sizet)
real :: zygrid(sizezy)
real :: salbgrid(sizesalb)
!real :: zelev(nbelts)
real :: marselev(nbelts)


data tempgrid/150., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., &
260., 270., 280., 290., 300., 310., 320., 330., 340., 350., 360., 370., &
     380., 390./ ! 25 temperatures

data pressgrid/1.e-5, 1.e-4, 1.e-3, 1e-2, 1.e-1, &
      1., 2., 3., 4., 5., 10., 15., &
      20., 25., 30., 35./ 	! 16 CO2 pressures 
	  
data zygrid/0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., & 
     55., 60., 65., 70., 75., 80., 85., 90./ ! 19 points

data salbgrid/0.0, 0.05, 0.1, 0.15, 0.2,0.25,0.3,0.35,0.4,0.45,  & 
     0.5,0.55, 0.6,0.65,0.7, 0.75,0.8,0.85,0.9,0.95,1./ ! 21 points 	 	  

!data zelev/ 2.272, 1.42, 0.388, 0.005, 0.005, 0.106, 0.156, 0.121, 0.154, & 
!0.158, 0.146, 0.366, 0.496, 0.382, 0.296, 0.202, 0.220, 0.137/ ! Earth elevation (in km)

!data marselev/2.8907, 1.6004, 1.2991, 0.7881, 0.3252, 0.7581, 1.4616, &    
!1.5818, 0.7754, -0.4717, -1.1372, -1.6280, -2.2597, -3.0221, -3.8864, &   
!-4.3499, -4.6942, -3.4123/ ! Mars elevation (in km), 18 latitude belts


data marselev/3.4220, 2.3636, 1.7409, 1.4600, 1.3027, 1.2955, 1.1241, &
      0.4522, 0.3017, 0.3487, 0.5717, 0.9445, 1.3362, 1.5870, 1.6930, &    
      1.4705, 1.0206, 0.5302, -0.1643, -0.7790, -1.1058, -1.1686, -1.4468, &   
      -1.8093, -2.1609, -2.3584, -2.7182, -3.3260, -3.7635, -4.0093, &   
      -4.2477, -4.4522, -4.6451, -4.7434, -4.0741, -2.6991/ ! Mars elevation (in km), 36 latitude belts


real*8 ecc, m, e

dimension :: latrad(0:nbelts + 1), lat(0:nbelts + 1), midlatrad(nbelts),&  
       midlat(nbelts), x(0:nbelts+1), focean(nbelts), delx(0:nbelts), &
	   h20alb(0:90), area(nbelts), temp(0:nbelts+1),  c(nbelts),  &
	   tprime(0:nbelts), tprimeave(nbelts), t2prime(nbelts),  &
	   ir(nbelts), mu(nbelts), s(nbelts), atoa(nbelts), surfalb(nbelts), &
       acloud(nbelts), zntempmin(nbelts), zntempmax(nbelts), &
	   zntempsum(nbelts), zntempave(0:nbelts+1), zndecmax(nbelts), &
	   zndecmin(nbelts), obstemp(nbelts), iceline(0:5), fice(nbelts), &
	   wthrate(nbelts), warea(nbelts), imco2i(nbelts), imco2l(nbelts),&
	   pco2(nbelts),zice(nbelts), icealb(nbelts), zicesum(nbelts), &
	   pco2r(nbelts), ph2o(nbelts), fco2(nbelts), psurf(nbelts), &
       cpco2(2), imco2s(nbelts), newice(nbelts), icealbco2(nbelts), CO2ps(nbelts)
	   
	   ! Arrays with nbelts +2 include 90 degree(nbelts+2) and -90  degree (1) end points(1 and nbelts+2).
	   
 character(len=18)::aa
 CHARACTER*5 :: STAR
 character  header*80

 
! INPUT FILE FLAGS----------------------------------------
inputebm = 1
oceandata = 2
fresneldata = 3

! OUTPUT FILE FLAGS
modelout = 4
seasonout = 5
 co2iceout = 6
 co2siceout = 10
 co2lout = 11
!----------------------------------------------------------

! READING INPUT FILE INITIAL PARAMETERS AND FLAGS
open(inputebm, file = 'input_ebm.dat', status = 'old')

do i = 1,2
read(inputebm,*) ! skips 2 lines before reading lines
enddo

read(inputebm,*)aa, seasonflag
read(inputebm,*)aa, igeog
read(inputebm,*)aa, icoldflag
read(inputebm,*)aa, Pdry
read(inputebm,*)aa, pco2i
read(inputebm,*)aa, lday
read(inputebm,*)aa, g
read(inputebm,*)aa, a0
read(inputebm,*)aa, ecc
read(inputebm,*)aa, peri
read(inputebm,*)aa, obl
read(inputebm,*)aa, ocean
read(inputebm,*)aa, groundalb
read(inputebm,*)aa, STAR
read(inputebm,*)aa, smass
read(inputebm,*)aa, tend

!-----------------------------------------------------
! READING OLR MATRIX


OLRparam = 7 ! unit number of OLR interpolation file


open(OLRparam, file = 'data/OLRparaminterpdry.325')  ! changed to hybrid files for both c-rr 6/19/2019


       read(OLRparam, *) ! skips 1 line
	   
	   do PP =  1, size(pressgrid)
	   
	      
		  do  TT = 1,size(tempgrid)
	          
	         read(OLRparam,*) aaa,bbb,ccc,ddd,olr(TT,PP),eee,fff
!			 write(40,*) olr(TT,PP), TT, PP
	   
	      enddo ! ends temperature loop
	   
	   enddo ! ends pressure loop


!---------------------------------------------------------------------
! READING PALB matrix	
PALBparam = 8 ! unit number of PALB interpolation file

if(STAR.eq.'F0')then
open(PALBparam, file = 'data/PALBparaminterpF0.325')

elseif(STAR.eq.'Sun')then
open(PALBparam, file = 'data/PALBparaminterpdry.325')! changed to hybrid files for both warm and cold starts c-rr 6/19/2019


elseif(STAR.eq.'K2')then
open(PALBparam, file = 'data/PALBparaminterpK2.325')
elseif(STAR.eq.'M0')then
open(PALBparam, file = 'data/PALBparaminterpM0.325')
elseif(STAR.eq.'M5')then
open(PALBparam, file = 'data/PALBparaminterpM5.325')
elseif(STAR.eq.'M8')then
open(PALBparam, file = 'data/PALBparaminterpM8.325')
endif


       read(PALBparam, *) ! skips 1 line

   do ZY = 1, sizezy
   	 do ALB = 1, sizesalb
   

	     do PP =  1, sizep
	   
	        do  TT = 1,sizet
	          
	      read(PALBparam,*)aaa,bbb,palb(TT,PP,ZY,ALB),ccc,ddd,eee,fff,ggg,hhh
	 
		  enddo ! ends temperature loop
	   
	   enddo ! ends pressure loop
	   
	 enddo ! ends surface albedo loop

   enddo ! ends zenith angle loop
   
!-------------------------------------------------------------
! READ TSTRAT matrix
TSTRATparam = 9 ! unit number of TSTRAT interpolation file

open(TSTRATparam, file = 'data/TSTRATparaminterp.325')



       read(TSTRATparam, *) ! skips 1 line
	   do PP =  1, sizep
	       do  TT = 1,sizet

		  	          
	       read(TSTRATparam,*) aaa,bbb,ccc,ddd,eee, fff, tempstrat(TT,PP),ggg
 
		   enddo ! ends temperature loop
	   
	      enddo ! ends pressure loop
	   
!----------------------------------------------------
if(icoldflag.eq.1)then 
		fcloud = 0.0 ! If cold start, no fractional cloud cover (from water vapor).. only that for CO2 calculated later in the code
        cloudir = 0.0  ! Following Turbet et al. (2017), there is no clouds in cold star case.
!        fracd0(1:nbelts) = 0.12/0.58  ! An arbitrarily low heat transfer for cold starts.. , which gives you a heat distribution similar to Turbet et al. (2017) Figure 1 cold start case and makes cloudir = 0.0.
        temp(0:nbelts+1) = 230.
        tempi = 230.
        landsnowfrac = 1.0 
elseif(icoldflag.eq.0)then
	    tempi = 280.
        temp(0:nbelts+1) = 280.  ! start at some beginning temperature if not snowball  = 280.
		fcloud = 0.26!0.037037037037036986*tempi - 10.1111111111111097!0.26  ! if warm start, then fractional cloud cover for water vapor clouds is 0.55
        cloudir = -6.0
!        fracd0(1:nbelts) = 1. ! Earth-like heat transfer... 
        landsnowfrac =0.5
endif

    
! initialize variables
CO2ps(1:nbelts) = 0.0
d0 = 0.58 !  thermal diffusion coefficient (default: 0.58).. 0.12 for Turbet et al. copy?

! ALLOW FRACDO TO EVOLVE NATURALLY
!fracd0(1:nbelts) = (0.12 + 0.0063013699*(tempi - 200.))/d0  
!if(tempi.ge.273.15)fracd0(1:nbelts) = 1d0
!if(tempi.le.200.)fracd0(1:nbelts)= 0.12d0/0.58d0 

a = a0*1.49597892E11 ! in meters
alpha = -0.078 !alpha: cloud albedo = alpha + beta*zrad..
beta = 0.65
cw = 2.1e8       !heat capacity over ocean
cl = 5.25e6      !heat capacity over land
 

v = 3.3e14 !volcanic outgassing for Earth (g/year) [from Holland (1978)].. should be scaled  for different planets (due to size, mass, radius, atmospheric composition, mantle fugacity...etc)
fdge = 1.64 ! fudge factor so that weathering rate = 3.3e14 g/yr for modern Earth conditions
wco2e = v ! carbonate-silicate cycle weathering rate, equal to volcanic outgassing rate in steady state.


avemol0 = 28.965*mp  !mean-molecular weight of present atmosphere (g)
hcp0 = 0.2401    !heat capacity of present atmosphere (cal/g K)
hcpco2 = 0.2105  !heat capacity of co2 (cal/g K)
hcph2o = 0.47    ! heat capacity of h2o (cal/g K)
last = 0       ! At last step of do while loop, this turns to "1"
prevtempave = 0.d0 ! initial previous temperature.
zrad = 0.d0
imco2s(1:nbelts) = 0
imco2i(1:nbelts) = 0
imco2l(1:nbelts) = 0
psat = 0.d0
!snowalb = 0.663  ! ..dirty snow.. changed this from 0.7 as per Caldiera & Kasting (JDH)

ann_tempave =  10. ! just some initial  value for annual temperature  average that  is greater than  zero.
ann_pco2ave = 0. ! initial  value for annual average pco2
 po = 1. ! total surface  pressure for the Earth (1 bar)
 rot0 = 7.27e-5 ! rotation  rate for Earth (rads/sec)
zntempmax(1:nbelts) = 0. ! initial guess on  max zonal temperature 
zntempmin(1:nbelts) = 500. ! initial guess on minimum  zonal temperature. An arbitarily high value.
zntempsum(1:nbelts) = 0. ! initial guess on zonal  temperature sum
albavesum = 0.d0  ! average albedo sum term initial value
fluxavesum = 0.d0 ! average flux sum term initial  value
iravesum = 0.d0   ! average outgoing infrared flux initial value
co2siceavesum = 0.d0 ! CO2 ice average sum initial value
co2iceavesum = 0.d0 ! CO2 ice average sum initial value
co2lavesum = 0.d0 ! CO2 clouds average sum initial value
wthratesum = 0.d0 ! weathering rate sum initial value
zicesum(1:nbelts) = 0.d0 ! zice sum inirialized.
decold = 0.d0  ! old declination value
decmid = 0.d0 ! mid declination value
decnew = 0.d0  ! new declination value
nfile = 0
nwthr = 0     ! seasonal/orbital counter
nstep = 0     ! time step  counter within an orbit.  Reset at the beginning of each new orbit.
! b = 1.
 compcnvg = 1. ! initial value greater than  cnvg
 


! INPUT DATA
! OPEN OCEAN DATA TABLE---------------------------------


open(oceandata, file = 'data/oceans.dat',status= 'old')



!---------------------------------------------------------
      ! Starting temperatures
      oblrad = obl*pi/180.

open(modelout, file ='out/model.out', status='unknown')
open(seasonout, file ='out/seasons.out', status= 'unknown')
open(co2iceout, file ='out/co2ice.out', status = 'unknown')
open(co2siceout, file ='out/co2sice.out', status = 'unknown')
open(co2lout, file ='out/co2l.out', status = 'unknown')

!  WRITE OBLIQUITY TO OUTPUT
      write(modelout,2) 'OBLIQUITY: ', obl, ' degrees'
 2    format(/ a,f5.2,a)

!  WRITE CO2-CLOUD FILE HEADER
      write(co2iceout,3)
	  write(co2siceout,3)
	  write(co2lout,3)
 3    format(58x,'IMCO2')
      write(co2iceout,33)
	  write(co2siceout,4)
	  write(co2lout,5)

 33    format(2x,'solar dec.(deg)',2x,'-88',1x,'-82',1x,'-77', &
     1x,'-72',1x,'-68',1x,'-62',1x,'-58',1x,'-52',1x,'-47',1x,'-43',1x, &
     '-37',1x,'-32',1x,'-28',1x,'-23',1x,'-18',1x,'-13',2x,'-7',2x,'-2', &
     3x,'2',3x,'7',2x,'12', &
     2x,'17',2x,'23',2x,'28',2x,'32',2x,'38',2x,'42',2x,'47',2x, &
     '52',2x,'58',2x,'63',2x,'68',2x,'72',2x,'78',2x,'82', 2x, '88', 5x, 'global CO2 ice cloud coverage' /)


 4    format(2x,'solar dec.(deg)',2x,'-88',1x,'-82',1x,'-77', &
     1x,'-72',1x,'-68',1x,'-62',1x,'-58',1x,'-52',1x,'-47',1x,'-43',1x, &
     '-37',1x,'-32',1x,'-28',1x,'-23',1x,'-18',1x,'-13',2x,'-7',2x,'-2', &
     3x,'2',3x,'7',2x,'12', &
     2x,'17',2x,'23',2x,'28',2x,'32',2x,'38',2x,'42',2x,'47',2x, &
     '52',2x,'58',2x,'63',2x,'68',2x,'72',2x,'78',2x,'82', 2x, '88', 5x, 'global CO2 surface ice coverage' /)

 5    format(2x,'solar dec.(deg)',2x,'-88',1x,'-82',1x,'-77', &
     1x,'-72',1x,'-68',1x,'-62',1x,'-58',1x,'-52',1x,'-47',1x,'-43',1x, &
     '-37',1x,'-32',1x,'-28',1x,'-23',1x,'-18',1x,'-13',2x,'-7',2x,'-2', &
     3x,'2',3x,'7',2x,'12', &
     2x,'17',2x,'23',2x,'28',2x,'32',2x,'38',2x,'42',2x,'47',2x, &
     '52',2x,'58',2x,'63',2x,'68',2x,'72',2x,'78',2x,'82', 2x, '88', 5x,'global liquid CO2 coverage' /)	 
	 


	 
! READ FRESNEL REFLECTANCE TABLE------------------------
open(fresneldata, file = 'data/fresnel_reflct.dat', status='old')
!open(fresneldata, file = 'data/briegleb_reflct.dat', status='old') ! fresnel data replacement (Briegleb et al. 1986; Enomoto et al. 2007)

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


!--------SET UP LATITUDINAL GRID (BELT BOUNDARIES)-----
!NEED ZERO INDEX FOR TPRIME and  TPRIME2 that uses  k-1 index

latrad(0) = -pi/2  !  -180 degrees(in radians)
lat(0) = latrad(0)*180./pi  ! -180 degrees


latrad(nbelts + 1) =  pi/2  !  -180 degrees(in radians)
lat(nbelts+1) = latrad(nbelts +1)*180./pi  ! -180 degrees

!  print *, lat(1), lat(nbelts+1)
  
! Creating 36 latitudinal bands 
do k = 1, nbelts, 1
  latrad(k) = latrad(0) + k*pi/nbelts     ! latitudinal (in rads)
  lat(k) = latrad(k)*180./pi ! latitudinal (in degrees)
  midlatrad(k) = latrad(0) + (1 + 2*(k-1))*pi/(2*nbelts)  !  center of latitudinal bands (in rads)
  midlat(k) = midlatrad(k)*180./pi !  center of latitudinal bands (in degrees)
enddo

! Fixing ..getting the right angles here for the beginning.
midlatrad(1) = -87.5*pi/180.
midlat(1) = -87.5 
 
 
! Latitudinal boundaries in sinusoidal space
x(0) = sin(latrad(0))
x(nbelts + 1) = sin(latrad(nbelts+1))

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

!---------------------------------------------------
! Initialize ice albedo  given star type


if(STAR.eq.'F0')then
Lum = 4.3
fvis = 0.64
fir = 0.36
elseif(STAR.eq.'Sun')then
Lum = 1.
fvis = 0.533
fir = 0.467
elseif(STAR.eq.'K2')then
Lum = 0.29
fvis = 0.34
fir = 0.66
elseif(STAR.eq.'M0')then
Lum = 0.08
fvis =0.127 
fir = 0.873
elseif(STAR.eq.'M5')then
Lum = 0.011
fvis = 0.054
fir = 0.946
elseif(STAR.eq.'M8')then
Lum = 0.0009
fvis = 0.0066
fir = 0.99339
endif

!--------------------------------------------------------
! BEGIN INTEGRATION  AT VERNAL EQUINOX. FIRST CALCULATE TIME SINCE PERIHELION.
solcon = Lum/(a0**2)  ! getting the stellar effective flux (seff)in units of what Earth receives at 1 AU (solcon = 1)
 
 
d = d0 ! initialize diffusion coefficient.. 

pco2r(1:nbelts) = pco2i ! initialize pco2. This is for OLR and PALB parameterizations only
!fco2(1:nbelts) = pco2i/((Pdry/28. + pco2i/44.)*44.) ! initial mixing ratio after diffusing over atmosphere. 
Ptot = Pdry + pco2i ! total dry initial pressure (bar)
fco2(1:nbelts) = pco2i/Ptot ! initial mixing ratio after diffusing over atmosphere (real fco2)... ignore water vapor in this first step
psurf(1:nbelts) = Ptot ! ignore water vapor contribution in first step..
pco2(1:nbelts) = Ptot*fco2(1:nbelts) ! the pco2 that you use for most of EBM except for OLR and PALB parameterizations

ann_pco2ave = pco2i

rot = (360./lday)*(pi/180.) ! rotation rate  (rad/secs)

mstar = smass*msun ! stellar mass (in kg)

w = ((grav*mstar)**0.5)/(a**1.5)  ! 2*pi/(orbital period(sec))
!print *, 2*pi/w  ! orbital period seconds

P = 2*pi/w  ! period for one orbit (seconds)
dt = lday/10 ! seconds per time step (default: 10 time steps per day)


dtlong = 300*dt ! time step is larger for the carbonate-silicate cycle (default 30: days) 
tlong = dtlong   ! first long time step

numsteps = tend/dt ! number of time steps in calculation

! HERE 3 arrays


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

! tcalc

	  orbitcount = 1 ! starting the very first orbit

!============BIG DO WHILE LOOP!!!!!!===================================================================================
	
    DO WHILE((tcalc.le.tend))
    
!	write(60,*)pco2(1:9), pco2r(1:9)

! CONVERGENCE MUST BE ACHIEVED  AND  MINIMUM NUMBER OF ORBITS TOO BEFORE LAST EQUALS ONE!
          if((compcnvg.le.cnvg).and.(orbitcount.ge.5))then  
!          if((compcnvg.le.cnvg))then  ! 
          last = 1  ! this becomes the last orbit
          tlast = 0
          endif

!          write(40,*)orbitcount,nstep
	
	sumsteps = sumsteps + 1
	
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
	        co2sicesum = 0.
	        co2icesum = 0.
            co2lsum = 0.
            pco2sum = 0.
			ph2osum = 0.
            psurfsum = 0.
                       fracd0sum = 0.
         else ! if no seasons
	  
!  ============ MEAN ANNUAL CALCULATION with variable obliquity from Ward. Integration is done in the insolation section and uses a function defined at the bottom of this code.
       q = q0*solcon

      if (tcalc.lt.tend)then
      t = t + dt
      tcalc = tcalc + dt
      tlast = tlast +  dt
	  
      else
         write(*,*) 'Calculation time has elapsed.'
         CALL WRAPUP(last,k,ann_tempave,ann_albave,ann_fluxave,ann_irave,ann_wthrateave, &
	  ann_pco2ave, ann_ph2oave,ann_psurfave,wco2e,d,pco2,nbelts,modelout,zntempmin, tcalc,orbitcount) 
      endif

	  
	  
! At the beginning of each time step in mean annual calculation, these sums are set to zero. Looking for sums of all Latitudes within a time step
            tempsum = 0. 
	        fluxsum = 0.
	        albsum = 0.
	        irsum = 0.
	        globwthrate = 0.
	        co2sicesum = 0.
	        co2icesum = 0.
            co2lsum = 0.
	        pco2sum = 0.
            ph2osum = 0.
			psurfsum = 0.
                        fracd0sum = 0.
      endif  ! ends season/no season if logic



!--- FINITE DIFFERENCING - ATMOSPHERIC and OCEANIC ADVECTIVE HEATING 

      do  k= 0,nbelts   !**first derivatives between grid points
         tprime(k) = (temp(k+1) - temp(k))/delx(k)
!		 if(sumsteps.eq.1)write(40,*) tprime(k), temp(k),  temp(k+1), delx(k)
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
                            
	          phi = log(pco2(k)/3.3e-4)   
	 
       fco2(k) = pco2(k)/psurf(k)   ! Changed from Ptot to psurf(k), which includes pH2O 
              
              if (fco2(k) .le. 1.e-1)then
       pco2r(k) = 44*Pdry*fco2(k)/(28*(1-fco2(k))) ! this is the pco2 array with no CO2 diffusion figured in. This is for PALB and OLR calculations only.	   
       else
                  pco2r(k) = pco2(k) ! at higher pressure CO2 atmospheres don't do the diffusion thing
                   endif

!       write(40,*)pco2r(k), temp(k)	
!==============================================================================   
! BILINEAR INTERPOLATION FOR CALCULATION OF OUTGOING INFRARED RADIATION FOR ALL STARS
! Valid for 150  < T < 390 K and 1e-5 < pco2 < 35 bar.

	   
       CALL INTERPOLR(temp(k), pco2r(k), TL, PL, TR, PR, & 
	   PLOGL, PLOGR, pco2rlog)
	   
	  Z0 = olr(TL, PR)
	  Z1 = olr(TR, PR)
	  Z2 = olr(TL, PL)
	  Z3 = olr(TR, PL)


      DEN = ((tempgrid(TR) - tempgrid(TL))*(PLOGR - PLOGL))	! Denominator to NA, NB, NC, and ND  
	  
	  NA =((tempgrid(TR) - temp(k))*(pco2rlog - PLOGL))/DEN
	  NB =((temp(k) - tempgrid(TL))*(pco2rlog-PLOGL))/DEN
	  NC =((tempgrid(TR)-temp(k))*(PLOGR-pco2rlog))/DEN
	  ND =((temp(k)-tempgrid(TL))*(PLOGR-pco2rlog))/DEN
      
!       write(40,*)pco2r(k), temp(k),Z0,  Z1, Z2, Z3, NA, NB, NC, ND, DEN
	  
      ir(k)  =   Z0*NA + Z1*NB + Z2*NC + Z3*ND 	
 
	  
!      	   write(40,*)k, temp(k), PLOGL, PLOGR, pco2r(k), pressgrid(PR), pressgrid(PL), Z4		  
!===============================================================================================	   
	   
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

! DETERMINE ICE FRACTION AS A FUNCTION OF  TEMPERATURE=====================================================

         if (temp(k).ge.273.15) then
            fice(k) = 0
         else if (temp(k).le.239.d0) then  ! Not le. 263.15. This is better in line with Thompson and Barron Table I (1981)
            fice(k) = 1
         else
            fice(k) = 1. - exp((temp(k)-273.15)/12.5) ! changed from divided by 10 to divided by 12.5 Better fit to Thompson and Barron Table I (1981)
         end if
		 
		 
! Temperature dependence of the albedo of snow/ice mixtures for both visible and infrared wavelengths (modified from Curry et al. 2001 and Barry et al. (1996) parameterizations)
if(temp(k).le.263.15)then
aice_vis = 0.7
elseif((temp(k).gt.263.15).and.(temp(k).lt.273.15))then
!aice_vis = 0.7 - 0.035*(temp(k) - 263.15)
aice_vis = 0.7 - 0.020*(temp(k) - 263.15)
else ! for 	T = 273.15	 
!aice_vis = 0.35	
aice_vis = 0.5		 
end if


if(temp(k).le.263.15)then
aice_nir = 0.5
elseif((temp(k).gt.263.15).and.(temp(k).lt.273.15))then
aice_nir = 0.5 - 0.028*(temp(k) - 263.15)
else ! for 	T = 273.15	 
aice_nir = 0.22		 
end if		 
		 
		 
! ALTERNATE VIS ICE ALBEDO FEEDBACK..		 
!aice_vis = aice_nir
		 
		 
		 
  snowalb = aice_nir*fir + aice_vis*fvis
  if(icealbco2(k).gt.0.1)snowalb = icealbco2(k)  ! If CO2 is condensing within latitude band, that becomes the new ice albedo for that band..c-rr 6/11/2019

!  snowalb = 1.e-6	 
		 
		 ! ocean albedo
		 if (temp(k).ge.273.15) then
		 oceanalb = h20alb(int(z))  ! ocean albedo 
		 elseif (temp(k).le.239.d0) then  ! Now ocean albedo has snow/ice albedo at temps below 239 K, not 263.15 K
		 oceanalb = snowalb
		 else ! between 239 and 273 K
		 oceanalb = h20alb(int(z))*(1. - fice(k)) + snowalb*fice(k)
		 endif
		 
		 
		 
!      end if   
!====================================================================
 
!if(sumsteps.eq.1)write(60,*)temp(k),  fice(k) 

      acloud(k) = alpha + beta*zrad
      if (acloud(k).le.0)acloud(k) = 0.1

!      write(40,*)acloud(k), zrad*180./3.14159
	  
!       focean(16:18) = 1.0
	  
      if (temp(k).le.263.15) then
!      if (temp(k).le.256.d0) then	  ! 256 K ensures that mean surface temperature below ~ 263 - 264 K, leads to total glaciation... for dense CO2 atmospheres
	  ! LAND WITH STABLE SNOW COVER; SEA-ICE WITH SNOW COVER
      landalb = snowalb*landsnowfrac + groundalb*(1 - landsnowfrac)
      icealb(k) = snowalb
	  
      ci = 1.05e7  ! thermal heat capacity of ice  (cl and  cw is the same for  land and water,  respectively)
      c(k) = (1-fice(k))*focean(k)*cw + fice(k)*focean(k)*ci + &
       (1 - focean(k))*cl
      fwthr = 0.  ! fraction of surface for weathering
	  
	  surfalb(k) = max(((1-focean(k))*landalb + &
      focean(k)*((1-fice(k))*oceanalb + fice(k)*icealb(k))), &
      fcloud*acloud(k))
	 
	  
      elseif(temp(k).ge.273.15)then ! if temperature are above 273.15 K then
      landalb = groundalb
      fice(k) = 0.
      fwthr = 1.       !**100% if T > 273K, 0% otherwise
      c(k) = focean(k)*cw + (1 - focean(k))*cl
	  
	  surfalb(k) = (1-fcloud)*((1-focean(k))*landalb + &  
      focean(k)*((1-fice(k))*oceanalb + fice(k)*icealb(k))) + &
      fcloud*acloud(k)     
	  
!	  elseif((temp(k).gt.256.d0).and.(temp(k).lt.273.15))then
	  elseif((temp(k).gt.263.15).and.(temp(k).lt.273.15))then
	  landalb = groundalb
!      fice(k) = 0.
      fwthr = 1. 
	  c(k) = focean(k)*cw + (1 - focean(k))*cl
	  
	  
	  surfalb(k) = max(((1-focean(k))*landalb + &
      focean(k)*((1-fice(k))*oceanalb + fice(k)*icealb(k))), &
      fcloud*acloud(k))
      endif 

 !=======================================================================     
      


!if(sumsteps.eq.1)write(60,*)focean(k),oceanalb,icealb,fice(k),fcloud      


      if(last.ne.1)then 
       CALL WEATHERING2(area, k, pco2, nbelts, focean, fwthr, & 
	   wthrate, warea, wco2,wco2e,v,dt, temp)
	   
!	   CALL WEATHERING(area, k, nbelts, focean, fwthr, wthrate, warea, temp)
      endif


!----------------------------------------------------------------------c
!  CO2 CLOUDS/ICE (Determines if atmospheric conditions allow for CO2 cloud/ice formation and says where they are)
!      if(last.eq.1)then    ! NOW ALWAYS CHECK FOR CO2 CLOUDS/ICE!!!
	  

	   
!	   if(last.eq.1)write(40,*)tstrat

	  K1 = tempstrat(TL, PR)
	  K2 = tempstrat(TR, PR)
	  K3 = tempstrat(TL, PL)
	  K4 = tempstrat(TR, PL)	  
	   
	   
      tstrat = NA*K1 + NB*K2 + NC*K3 + ND*K4 
	  
!     if(last.eq.1)print *, 'tstrat=', tstrat, k
	 
	 
	 
      fn2 = Pdry/Ptot
      cpco2(1) = -0.000000004052836*tstrat**3  + 0.000002828904429*tstrat**2 +  0.000452556332556*tstrat  + 0.564411188811185  ! specific heat CO2 at stratosphere
      cpco2(2) = -0.000000004052836*temp(k)**3 + 0.000002828904429*temp(k)**2 + 0.000452556332556*temp(k) + 0.564411188811185  ! specific heat CO2 at surface
	  cpn2 = 1.04 ! specific heat of N2
	  
	  cptot = fn2*cpn2 + fco2(k)*0.5*(cpco2(1)+cpco2(2)) ! total specific heat
	  
	  gamd=g/cptot  ! dry lapse rate


!======================================================================================	   	   
!
      nh = 0
	  tlapse =  0.5*gamd ! assume lapse rate is half dry lapse rate (relatively intermediate value close to both critical and moist lapse rates. Stone and Carlson (1979) For Earth, close to average!)
	  
!     if(last.eq.1)print *, 'tlapse=', tlapse, k
	  
	  htemp = temp(k) - tlapse*nh
	  
	  if (htemp.lt.tstrat)then   ! which skips the entire do while loop condition  BELOW because CO2 clouds/ice do not form with this condition
	  
      CALL WEATHERING2(area, k, pco2, nbelts, focean, fwthr, & 
	   wthrate, warea, wco2,wco2e,v,dt, temp)
	   
!      CALL WEATHERING(area, k, nbelts, focean, fwthr, wthrate, warea, temp)

      endif ! ends htemp/tstrat if logic
	  
!	  	  write(60,*) avemol, htemp,tstrat,nh
!=============================================================================	  


	  do while(htemp.ge.tstrat)

	  

      hpco2 = pco2(k)*((temp(k)-tlapse*nh)/temp(k))** &
       (avemol*100*g/(1.38e-16*tlapse*1.e-5))
	   
	   
!       if(last.eq.1)write(40,*)hpco2, psat, htemp, imco2l(k), imco2i(k) Why is imco2l flag not changing to 1?

	        ! SUBROUTINE SATCO2 from Jim Kasting "earthtem_0.f"
			
	        !   VAPOR PRESSURE OVER LIQUID...According to Co2 phase diagram, CO2 clouds would form under  these conditions
	        if (htemp.gt.216.56)then 
            psl = 3.128082 - 867.2124/htemp + 1.865612e-2*htemp - &
            7.248820e-5*htemp**2 + 9.3e-8*htemp**3
			

            !  VAPOR PRESSURE OVER SOLID ..According to CO2 phase diagram, CO2 ice only would form under these condition
            elseif (htemp.lt.216.56)then  

           ! (Altered to match vapor pressure over liquid at triple point)
            psl = 6.760956-1284.07/(htemp-4.718) + 1.256e-4*(htemp-143.15)
			


            endif 
			
			patm = 10.**psl
            psat = 1.013*patm

			
        imco2i(k) = 0  ! only atmospheric co2 ices need to be initialized after each time step..c-rr 6/5/2019
!        imco2s(k) = 0 
!	imco2l(k) = 0
	
!        if(last.eq.1)write(40,*)(hpco2 - psat), (htemp - 216.56)		! It looks like CO2 clouds are NEVER formed!!! 
		
        if ((hpco2.ge.psat).and.(htemp.lt.216.56)) then      !**co2 ices form
        imco2i(k) = 1  ! this labels which latitude CO2 ices form. 
        
        write(40,*)hpco2,psat

		 	
	  
		
        CALL WEATHERING2(area, k, pco2, nbelts, focean, fwthr, & 
	   wthrate, warea, wco2,wco2e,v,dt, temp)
	   

        exit ! THIS do while loop is  exited 
		
        elseif (hpco2.lt.psat)then ! if hpco2 lt psat then ices do not form and need to go to a higher (cooler) altitude (1 km higher) and check again
		nh = nh + 1
	    htemp = temp(k) - tlapse*nh
		
!	     if(last.eq.1)write(40,*)htemp			
!		     write(60,*) hpco2,psat,htemp,tstrat,nh,lat(k),imco2(k)
	      if (htemp.lt.tstrat)then
        CALL WEATHERING2(area, k, pco2,nbelts, focean, fwthr, & 
	   wthrate, warea, wco2,wco2e,v,dt, temp)
	   
!	   CALL WEATHERING(area, k, nbelts, focean, fwthr, wthrate, warea, temp)
	      exit  ! THIS do while loop is  exited 
          endif 
	  
         elseif ((hpco2.ge.psat).and.(htemp.gt.216.56).and.(hpco2.ge.5.1)) then  !** liquid co2 forms so long as PCO2 gt. 5.1 bar and above psat curve c-rr 6/5/2019
	     imco2l(k) = 1
!	     write(40,*)imco2l(k)
	 
	     CALL WEATHERING2(area, k, pco2, nbelts, focean, fwthr, & 
	       wthrate, warea, wco2,wco2e,v,dt, temp)

       exit   ! THIS do while loop is exited		  
		  
        endif  ! end if condition for CO2 cloud formation
		
		

       enddo ! ends do  while of htemp and tstrat loop!!
	  
!=============================================================================	  
	   
!       LOGIC FOR CO2 ICE CONDENSATION ON SURFACE



	    !   VAPOR PRESSURE OVER LIQUID...According to Co2 phase diagram, liquid CO2 would form under  these conditions
	    if (temp(k).gt.216.56)then 
            psl = 3.128082 - 867.2124/temp(k) + 1.865612e-2*temp(k) - 7.248820e-5*temp(k)**2 + 9.3e-8*temp(k)**3
			

            !  VAPOR PRESSURE OVER SOLID ..According to CO2 phase diagram, CO2 ice clouds only would form under these condition
            elseif (temp(k).lt.216.56)then  

           ! (Altered to match vapor pressure over liquid at triple point)
            psl = 6.760956-1284.07/(temp(k)-4.718) + 1.256e-4*(temp(k)-143.15)
			
            endif 
			
	    patm = 10.**psl
        psat = 1.013*patm
         		
!		write(40,*)delp, pco2(k), psat, k

        if ((pco2(k).ge.psat).and.(temp(k).lt.216.56)) then      !**co2 ices form on the surface
        imco2s(k) = 1  ! this labels which latitude CO2 ices form. 

         write(40,*)pco2(k), psat		
		
!   THIS CO2 ICE LOGIC IS COMMENTED OUT FOR NOW

!                newice(k) = pco2(k) - psat - delp  ! added newice, subtracting out old ice from previous time step..	
                newice(k) = pco2(k) - psat	
                if(newice(k).lt.0)newice(k) = 0.0 ! c-rr 6/5/2019
!		delp = pco2(k) - psat ! bars ! old ice from previous time step
		zice(k) = 101325.*newice(k)/(g*1600.)  ! thickness of CO2 ice formed in latitude band(in meters). 1600 kg/m^3 is the density of dry ice
		zicesum(k) = zice(k) + zicesum(k)
		icealbco2(k) = 0.6 ! 0.35 is the albedo for CO2 ice (Warren et al. 1990), replacing that of water ice
               
		pco2(k) = pco2(k) -  newice(k)  ! calculate new pco2 at latitude band

             CALL WEATHERING2(area, k, pco2, nbelts, focean, fwthr, & 
	       wthrate, warea, wco2,wco2e,v,dt, temp)
                         
           elseif ((pco2(k).ge.psat).and.(temp(k).ge.216.56).and.(pco2(k).ge.5.1)) then  !** liquid co2 forms on the surface..so long as PCO2 gt. 5.1 bar and above psat curve c-rr 6/5/2019

	   
	     CALL WEATHERING2(area, k, pco2, nbelts, focean, fwthr, & 
	       wthrate, warea, wco2,wco2e,v,dt, temp)

           elseif ((pco2(k).lt.psat).and.(temp(k).ge.216.56).and.(zicesum(k).gt.150.))then ! any liquid evaporates back into the atmosphere, so long as liquid CO2 glaciers form above 150m thick CO2 ice threshold. c-rr 6/5/2019

                 liql = (zicesum(k)- 150.)*(g*1600.)/101325. ! remainder melts off glacier and CO2 pressure that evaporates back up to atmosphere (in bar)  
                 pco2(k) = pco2(k) +  liql  ! calculate new pco2 at latitude band
                 

          elseif ((pco2(k).lt.psat).and.(temp(k).lt.216.56).and.(zicesum(k).gt.0.1))then ! ice sublimates back into atmosphere as long as there is some surface ice.. c-rr 6/15/2019
                subl = zicesum(k)*(g*1600.)/101325 ! in bar
                subl = min(subl, pco2i - pco2(k)) ! doesn't allow it to sublimate so much ice into the atmosphere so that it exceeds the original amount the atmosphere already had..
                pco2(k) = pco2(k) +  subl  ! calculate new pco2 at latitude band
                zicesum(k) = 0.0 ! reset ice thickness in this latitude band to zero. It's all been sublimated 



!                 write(40,*), subl, zicesum(k), pco2(k), psat, temp(k), k

	     CALL WEATHERING2(area, k, pco2, nbelts, focean, fwthr, & 
	       wthrate, warea, wco2,wco2e,v,dt, temp)

               endif
                
!========================================================================================
           ! If there is no surface ice accumulation in given band, then turn CO2 surface ice flag in that band to "zero." c-rr 6/6/2019
            if(imco2s(k).eq.1)then
                if(zicesum(k).le.0)imco2s(k) = 0
            endif

	 
!=====================================================================================	
                
	   ! QUADRILINEAR INTERPOLATION FOR PLANETARY ALBEDO. 
       ! Valid for 10^-5 < pco2 < 35 bars, 150 < T < 390 K, 0.0 < surfalb < 1, and 0 < z < 90 degrees for the Sun. 	   
       as = surfalb(k)  
	   
!	   if(k.eq.1)write(40,*)as, icealb(k), temp(k), k
	   
       CALL INTERPPALB(as,mu(k), ZYL, ZYR, zyy, AL, AR)	  
	  
	   
	   DEN = (tempgrid(TR) - tempgrid(TL))*(PLOGR - PLOGL)*(zygrid(ZYR) - zygrid(ZYL))*(salbgrid(AR) - salbgrid(AL))
	   
	  ! X0 = tempgrid(TL)        ! Y0 = PLOGL           !Z0 = zygrid(ZYL)      !T0 = salbgrid(AL)
	  ! X1 = tempgrid(TR)        ! Y1 = PLOGR           !Z1 = zygrid(ZYR)      !T1 = salbgrid(AR)
	  ! X2 = temp(k)             ! Y2 = pco2rlog	    !Z2 = zyy              !T2 = as	   

	   N_1 = (tempgrid(TR) - temp(k))*(PLOGR - pco2rlog)*(zygrid(ZYR) - zyy)*(as - salbgrid(AL))/DEN
	   N_2 = (temp(k) - tempgrid(TL))*(PLOGR - pco2rlog)*(zygrid(ZYR) - zyy)*(as - salbgrid(AL))/DEN
	   N_3 = (temp(k) - tempgrid(TL))*(pco2rlog - PLOGL)*(zygrid(ZYR) - zyy)*(as - salbgrid(AL))/DEN
	   N_4 = (temp(k) - tempgrid(TL))*(pco2rlog - PLOGL)*(zyy - zygrid(ZYL))*(as - salbgrid(AL))/DEN
	   N_5 = (tempgrid(TR) - temp(k))*(pco2rlog - PLOGL)*(zyy - zygrid(ZYL))*(as - salbgrid(AL))/DEN  
	   N_6 = (tempgrid(TR) - temp(k))*(PLOGR - pco2rlog)*(zyy - zygrid(ZYL))*(as - salbgrid(AL))/DEN   
	   N_7 = (tempgrid(TR) - temp(k))*(PLOGR - pco2rlog)*(zyy - zygrid(ZYL))*(salbgrid(AR) - as)/DEN     
	   N_8 = (temp(k) - tempgrid(TL))*(PLOGR - pco2rlog)*(zyy - zygrid(ZYL))*(salbgrid(AR) - as)/DEN     
	   N_9 = (temp(k) - tempgrid(TL))*(pco2rlog - PLOGL)*(zyy - zygrid(ZYL))*(salbgrid(AR) - as)/DEN     
	   N_10 = (temp(k) - tempgrid(TL))*(pco2rlog - PLOGL)*(zygrid(ZYR) - zyy)*(salbgrid(AR) - as)/DEN    
	   N_11 = (temp(k) - tempgrid(TL))*(PLOGR - pco2rlog)*(zygrid(ZYR) - zyy)*(salbgrid(AR) - as)/DEN    
	   N_12 = (tempgrid(TR) - temp(k))*(PLOGR - pco2rlog)*(zygrid(ZYR) - zyy)*(salbgrid(AR) - as)/DEN    
	   N_13 = (tempgrid(TR) - temp(k))*(pco2rlog - PLOGL)*(zygrid(ZYR) - zyy)*(salbgrid(AR) - as)/DEN    
	   N_14 = (tempgrid(TR) - temp(k))*(pco2rlog - PLOGL)*(zyy - zygrid(ZYL))*(salbgrid(AR) - as)/DEN    
	   N_15 = (tempgrid(TR) - temp(k))*(pco2rlog - PLOGL)*(zygrid(ZYR) - zyy)*(as - salbgrid(AL))/DEN    
	   N_16 = (temp(k) - tempgrid(TL))*(PLOGR - pco2rlog)*(zyy - zygrid(ZYL))*(as - salbgrid(AL))/DEN   
	   

	  
	  
	  !palb matrix needs to be read in..(temp, pressure, zenith angle, surface albedo)
	  K1 = palb(TL, PL, ZYL, AR)
	  K2 = palb(TR, PL, ZYL, AR)
	  K3 = palb(TR, PR, ZYL, AR)
	  K4 = palb(TR, PR, ZYR, AR)
	  K5 = palb(TL, PR, ZYR, AR)
	  K6 = palb(TL, PL, ZYR, AR) 
	  K7 = palb(TL, PL, ZYR, AL)
	  K8 = palb(TR, PL, ZYR, AL)
	  K9 = palb(TR, PR, ZYR, AL)
	  K10 = palb(TR, PR, ZYL, AL)
	  K11 = palb(TR, PL, ZYL, AL)
	  K12 = palb(TL, PL, ZYL, AL)
	  K13 = palb(TL, PR, ZYL, AL)
	  K14 = palb(TL, PR, ZYR, AL)
	  K15 = palb(TL, PR, ZYL, AR)
	  K16 = palb(TR, PL, ZYR, AR)

	  atoa(k) = N_1*K1 + N_2*K2 + N_3*K3 + N_4*K4 + N_5*K5 + N_6*K6 +N_7*K7 + N_8*K8 + & 
	  N_9*K9 + N_10*K10 + N_11*K11 + N_12*K12 + N_13*K13 + N_14*K14 + N_15*K15 + &
	  N_16*K16



!==========================================================================================

	  
	  
!if(sumsteps.eq.1)write(60,*)temp(k), atoa(k),as,pco2   

!----------------------------------------------------------------------c
!  DIURNALLY-AVERAGED INSOLATION 
      
      if (seasonflag.eq.1) then
         s(k) = (q/pi)*(x(k)*sin(dec)*h + &   !essentially eqn a8 in williams/kasting
             cos(asin(x(k)))*cos(dec)*sin(h))
		  
		 
!           if(sumsteps.eq.1)write(60,*)s(k), x(k),dec,h		  
		  	 
			 
      else ! if no seasons 
         ! for mean annual do the Ward insolation integration (JDH)
         call qtrap(sumsteps,0, twopi, s(k), latrad(k), oblrad)
         s(k) = (q/(2*(pi**2))) * 1/(sqrt(1 - ecc**2)) * s(k)
         !print *, 'insolation at', lat(k), ':', s(k)
      end if
!
!----------------------------------------------------------------------c
!  SURFACE TEMPERATURE - SOLVE ENERGY-BALANCE EQUATION 

               fracd0(k) = 1.  ! old constant d assumption

                      ! 23.5 degree obliquity
!                                              if (obl.eq.23.5)then
!                      fracd0(k) = 1.25604*mu(k) + 0.399734     ! from Vladilo et al. (2013) approach..scaled so that annual fracd0 = 1. Assuming equator 3 times better at heating than pole. OLR error 10 W/m^2
!                       fracd0(k) = 0.89734*mu(k) + 0.57115691    ! from Vladilo et al. (2013) approach..scaled so that annual fracd0 = 1. Assuming equator 2 times better at heating than pole. OLR error 6.5 W/m^2
!                        endif

!                         ! 45.0 degrees
!                      fracd0(k) = 1.2706*mu(k) + 0.4488     ! from Vladilo et al. (2013) approach..scaled so that annual fracd0 = 1. Assuming equator 3 times better at heating than pole. OLR error 10 W/m^2
!                       fracd0(k) = 0.8770*mu(k) + 0.6196    ! from Vladilo et al. (2013) approach..scaled so that annual fracd0 = 1. Assuming equator 2 times better at heating than pole. OLR error 6.5 W/m^2
     
                       ! 0 degree obliquity
!                                                if(obl.eq.0)then
!                       fracd0(k) = 1.2227*mu(k) + 0.3888     ! from Vladilo et al. (2013) approach..scaled so that annual fracd0 = 1. Assuming equator 3 times better at heating than pole. OLR error 10 W/m^2
!                       fracd0(k) = 0.8804*mu(k) + 0.5599    ! from Vladilo et al. (2013) approach..scaled so that annual fracd0 = 1. Assuming equator 2 times better at heating than pole. OLR error 6.5 W/m^2
!                                              endif                                                        
                 

      avemol = mp*(28.*Pdry + 44.*ann_pco2ave)/(Pdry+ ann_pco2ave) ! molecular weight expression
!      avemol = mp*(28.965*Pdry + 44.*ann_pco2ave)/(Pdry+ ann_pco2ave) ! molecular weight expression for Earth air instead.
      hcp = (hcp0 + ann_pco2ave*hcpco2)/(Pdry+ann_pco2ave)  ! heat capacity
      d = fracd0(k)*d0*((Pdry+ ann_pco2ave)/po)*((avemol0/avemol)**2)*(hcp/hcp0)* &   
        (rot0/rot)**2


      temp(k) = (d*t2prime(k)-ir(k)+s(k)*(1-atoa(k)))*dt/c(k) + temp(k)  !main energy balance equation
      
!     CALCULATING PARTIAL PRESSURE OF WATER VAPOR FOR FULLY-SATURATED ATMOSPHERES	
      L = 45180 - 42.84*(temp(k) - 273.15) ! J/mol.. converted from J/g from Stone and Carlson (1979)
      
	  ph2o(k) = po_star*exp(-L/(Rgas*temp(k)))/(101325.) ! average H2O surface pressure at each latitude (in bar)

         if(ann_tempave.lt.273.)ph2o(k) = 1.e-20 ! c-rr 6/19/2019

	  psurf(k) = Pdry + ph2o(k) + pco2(k) ! average total surface pressure at each latitude (in bar)
  
	  
	  
	  
	  
	  
	  
! PERHAPS ALSO PUT PCO2 evolution here but for SLOWER TIME STEP?? EVERY 5 ORBITS PERHAPS???? PERHAPS ADD PCO2 based on difference  between W and V?? I think so. And weathering (W) WAY outpaces CO2 volcanic outgassing at higher PCO2.. Perhaps while keeping temperature convergence?

	  
	  
!      if(last.eq.1)write(40,*)zntempmin(1:nbelts),k,last
	  
!if(sumsteps.eq.1)write(60,*)temp(k),  ir(k), s(k), atoa(k), c(k), d, t2prime(k)     
	  

!  SUM FOR GLOBAL AVERAGING  (WITHIN A GIVEN TIME STEP.. SUMMING AREA CONTRIBUTION OVER ALL LATITUDES TO GET GLOBAL IR, FLUX, PALB, TEMP..etc)
      irsum = irsum + area(k)*ir(k)
      fluxsum = fluxsum + area(k)*s(k)
      albsum = albsum + area(k)*s(k)*atoa(k)  ! albsum is flux upward, reflected to space (s(k)*atoa(k) scaled by area
      tempsum = tempsum + area(k)*temp(k)
      globwthrate = globwthrate + wthrate(k)

	  co2sicesum = co2sicesum + area(k)*imco2s(k)
      co2icesum = co2icesum + area(k)*imco2i(k)
	  co2lsum = co2lsum + area(k)*imco2l(k)

!   write(40,*)co2sicesum, area(k), imco2s(k)
        

      pco2sum = pco2sum + area(k)*pco2(k)
      ph2osum = ph2osum + area(k)*ph2o(k)	  
	  psurfsum = psurfsum + area(k)*psurf(k)
	  fracd0sum = fracd0sum + area(k)*fracd0(k)  
!	  if(k.eq.1)write(40,*)pco2sum, pco2(k), k
         
!	  if(sumsteps.eq.1)write(60,*)irsum, tempsum, fluxsum, k
           

      if(last.eq.1)then !  ZONAL STATISTICS - if last ORBIT====================================
 

      zntempmin(k) = amin1(zntempmin(k),temp(k))
	  
      if (zntempmin(k).eq.temp(k))then 
	  zndecmin(k) = decangle
      endif
      zntempmax(k) = amax1(zntempmax(k),temp(k))
	  
      if (zntempmax(k).eq.temp(k))then 
      zndecmax(k) = decangle
      endif  
      zntempsum(k) = zntempsum(k) + temp(k)

	  ! With this logic, zntempmin and zntempmax BOTH start out the same
!	      if(k.eq.1)write(40,*)temp(k), zntempsum(k),t, orbitcount  
	  
      endif  ! ends if logic for zonal statistics===========================================

CO2ps(k) = pco2i - pco2(k) ! CO2 pressure on the surface for given latitude for given time step
sumnewice = sumnewice + CO2ps(k)*area(k) !total amount of ice
!sumnewice = sumnewice/pco2i

       ENDDO  
!======**end of BIG belt loop=================================================  
write(50,*)sumnewice,tcalc/3.15e7
sumnewice = 0.   
       
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

!  CO2-CLOUD/ICE DATA
        if(last.eq.1)write(co2iceout,630) decangle,imco2i(1:nbelts),co2icesum
        if(last.eq.1)write(co2siceout,630) decangle,imco2s(1:nbelts),co2sicesum
        if(last.eq.1)write(co2lout,630) decangle,imco2l(1:nbelts),co2lsum
         

 630  format(3x,f6.2,9x,36(3x,i1),9x,f7.3)
 
!      write(60,*)decangle, co2cldsum, fluxsum  ! still 2.e4
	  
      endif !! ends last twrite logic SEASONS IF STUFF========================================
!----------------------------------------------------------------------c


! COME BACK FOR PCO2EVOLUTION!!!!!!!!!!!!!!!!!!!======================================
!	        if(tcalc.eq.tlong)then
!       pco2(1:nbelts) = ann_pco2ave + (((v - wthr)/P)*dtau)/5.1e21  ! Dividing by P to convert g/yr for v and w to g/sec.. 5.1e21 g in one bar Gets new pco2 array for next orbit..	  


!	  write(40,*)tcalc, dtlong, tlong, v, wthr, orbitcount
!	  tlong = tlong + dtlong ! increment dtlong step for next time
!            endif   
!=====================================================================================


!  GLOBAL AVERAGING (Done every time step)



     ! The global sum values over all latitudes for a given time step = average global values for a given time step 
      irave = irsum
      fluxave = fluxsum
      co2siceave = co2sicesum
      co2iceave = co2icesum
      co2lave = co2lsum

      albave = albsum/fluxave   ! flux up/flux down is albedo average
      tempave = tempsum
	  pco2ave = pco2sum
	  ph2oave = ph2osum
	  psurfave = psurfsum
          fracd0ave= fracd0sum


      ! summing all of these global averages over ALL time steps within each orbit
      iravesum = iravesum + irave
      fluxavesum = fluxavesum + fluxave
      albavesum = albavesum + albave
      tempavesum = tempavesum + tempave
      wthratesum = wthratesum + globwthrate
      co2siceavesum = co2siceavesum + co2siceave
      co2iceavesum = co2iceavesum + co2iceave
      co2lavesum = co2lavesum + co2lave
	  pco2avesum = pco2avesum + pco2ave  
	  ph2oavesum = ph2oavesum + ph2oave
	  psurfavesum = psurfavesum + psurfave
          fracd0avesum = fracd0avesum + fracd0ave 
      nstep = nstep + 1     ! Go to next time step within orbit. Nstep goes back to ZERO after each revolution around star
	  
!====================================================================================================================================================================
!====================================================================================================================================================================

      if(t.lt.P)then    !**one orbit since last averaging.. do nothing more and go to the end to restart the loop


		
!  ANNUAL AVERAGING (comes here after the end of each orbit=====================================================
      elseif (t.ge.P)then !**one orbit since last averaging.
      ann_tempave = tempavesum/nstep
      ann_albave = albavesum/nstep
      ann_fluxave = fluxavesum/nstep    ! annual averaging occurs after ONE full stellar revolution
      ann_irave = iravesum/nstep
      ann_wthrateave = wthratesum/nstep
      ann_co2siceave = co2siceavesum/nstep
      ann_co2iceave = co2iceavesum/nstep
      ann_co2lave = co2lavesum/nstep
    

                    if(ann_tempave.lt.273.)then  
                    fcloud = ann_co2iceave
                    cloudir = 0.0
!                    fracd0(1:nbelts) = 0.12/0.58
                    landsnowfrac = 0.5   !was 1 before
                    endif
                    
                    if (ann_tempave.lt.263.)landsnowfrac=1.0


                  if(ann_tempave.ge.273.)then  
                  fcloud = 0.55
                  cloudir = -6.0
!                  fracd0(1:nbelts) = 1.
                 landsnowfrac = 0.5
                  endif
                  
    !fracd0 evolves naturally
	! ALLOW FRACDO TO EVOLVE NATURALLY
	!fracd0(1:nbelts) = (0.12 + 0.0063013699*(ann_tempave - 200.))/d0  
	!if(ann_tempave.ge.273.15)fracd0(1:nbelts) = 1d0
	!if(ann_tempave.le.200.)fracd0(1:nbelts)= 0.12d0/0.58d0 
	
	if ((ann_tempave.ge.273.15).and.(ann_tempave.lt.288)) fcloud = 0.037037037037036986*ann_tempave -10.111111111111097
	if (ann_tempave.ge.288) fcloud = 0.55

      ann_pco2ave = pco2avesum/nstep 
	  ann_ph2oave = ph2oavesum/nstep
	  ann_psurfave = psurfavesum/nstep
      ann_fracd0ave = fracd0avesum/nstep
     print *, ann_fracd0ave   
	  
!====================================================================================================================================	  

		 if(last.eq.1)exit ! if last is 1 when it comes here a second time, EXIT the DO WHILE loop and write out the ZONAL STATISTICS.
		 
!====================================================================================================================================


!  ADJUST PCO2 EVERY 5 ORBITAL CYCLES.. WHY IS THIS LOGIC DONE????
!---------------------------------------	  	  
      if(nwthr.ge.5)then
      nwthr = 0 ! after 5th orbit, restart counter to  0
      wthr = fdge*wco2e*ann_wthrateave ! gives the actual global weathering rate (wco2) in g/yr.. (wthr = wco2 = wco2e*wco2/wco2e) with fudge factor (fdge) included
      
!	  write(40,*)wthr,  fdge, wco2e, ann_wthrateave, ann_pco2ave
	  
	   ! Don't let pcO2 get below 1.e-5 bar or above  35 bar,  which are the code limits.

!	   pco2(1:nbelts) = amax1(pco2(1:nbelts),1.e-5)
!	   pco2(1:nbelts) = amin1(pco2(1:nbelts),10.)
!	   write(40,*)orbitcount, ann_pco2ave, pco2(1:nbelts)
!       write(40,*)orbitcount, wthr, v  
      if (wthr.lt.v)then
      if(wthr.lt.1.e-2) then
	  wthr=1.

	  
	  
!	   dtau = 6*P ! after 5 orbit cycles (in seconds)
!       pco2(1:nbelts) = pco2avesum/nstep + ((v - wthr)/P)*dtau  ! Dividing by P to convert g/yr for v and w to g/sec.. Gets new pco2 array for next orbit..	  

      endif  !if wthr lt1.e-2
      endif ! if wthr lt.v
      endif ! if nwthr ge. 5

      if(nwthr.lt.5)then
	  nwthr = nwthr + 1   ! counter goes up every season  (orbit)
      endif ! END SEASONAL CYCLE PCO2 IF LOGIC
!-------------------------------------------

!=============================================================	  




       
	    compcnvg =abs(prevtempave-ann_tempave) ! compare with convergence criteria
        prevtempave = ann_tempave 
		! GET READY TO  SET LAST eq. 1.. WILL  STOP ONLY AFTER ORBITCOUNT IS REACHED
                         
           

 !        write(40,*) orbitcount, abs(prevtempave-ann_tempave)



      fluxavesum = 0.  ! all of these are reset before NEXT (FINAL) orbit (except co2cldavesum) in order to get zonal statistics
      albavesum = 0.
      tempavesum = 0.
      iravesum = 0.
      wthratesum = 0.
	  pco2avesum  = 0.
	  ph2oavesum = 0.
	  psurfavesum = 0.
          co2iceavesum = 0.
         co2siceavesum = 0.
         co2lavesum = 0.
         fracd0avesum = 0.
      t = 0
	  nstep = 0  ! at last orbit don't reset nstep to get zntemp calc below

      orbitcount = orbitcount + 1   ! go to next orbit
      
	  

      endif  ! ends t P if logic 


	  
       END DO  ! ends BIG DO  WHILE LOOP
!=======================================================================================================
	   
	   
	      if (last.eq.1) then  ! once everything else is done come here and  finishing write up SUMMARY AND FINAL ZONAL  STATS

!                               sumd0 = sum(fracd0(1:nbelts))
!                              print *, 'AVERAGE FRACD0 =', sumd0/nbelts
!                      d = (sumd0/nbelts)*d0*((Pdry+ ann_pco2ave)/po)*((avemol0/avemol)**2)* (hcp/hcp0)*(rot0/rot)**2			  
	  
	      CALL WRAPUP(last, k,ann_tempave, ann_albave, ann_fluxave, ann_irave, ann_wthrateave, &
	  ann_pco2ave, ann_ph2oave, ann_psurfave, wco2e,d, pco2, nbelts, modelout,zntempmin, tcalc, orbitcount)
	  
		  
		  
          CALL ZONALEND(shtempave, nhtempave, nbelts, zntempsum, &
		  zntempave, nstep, k, icelat, iceline, nedge, lat, modelout, &
		  zndecmin, zndecmax,zntempmax, zntempmin, ann_co2iceave, igeog, &
		   focean,ann_co2lave, coastlat, ocean, seasonout, midlat,x, ann_pco2ave, pco2i)

             if(ann_pco2ave.lt.pco2i)print *, 'SURFACE CO2 CONDENSATION HAS OCCURRED.'


	      endif ! ENDS ZONAL STATISTICS WRITING SECTION
					   
!          if(last.eq.1)print*, zicesum(1:nbelts)
!          if(last.eq.1)print*, pco2(1:nbelts) 
!          if(last.eq.1)print*, temp(1:nbelts) 
           

     	  if(last.eq.1)stop ! after convergence has been achieved stop code
		  


       END ! Ends program
!-------------------------------------------------------------------------------------------
!-------------------------------------------------------------------------------------------

       SUBROUTINE ZONALEND(shtempave, nhtempave, nbelts, zntempsum, &
	   zntempave, nstep, k, icelat, iceline, nedge, lat, modelout, &
	   zndecmin, zndecmax,zntempmax, zntempmin, ann_co2iceave, igeog, &
	   focean,ann_co2lave, coastlat, ocean, seasonout, midlat,x, ann_pco2ave, pco2i)
	   
       IMPLICIT NONE
		  
       integer :: nbelts, k, igeog, nstep, nedge, modelout, seasonout
		  
       real*4 :: shtempave, nhtempave, zntempsum, zntempave, icelat, &
	   iceline, lat, zndecmin, zndecmax, zntempmax, zntempmin, &
	   ann_co2lave, ann_co2iceave, focean, coastlat, ocean, midlat, x, ann_pco2ave, pco2i
		  
		  
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
         do k = 1,nbelts,1
         write(seasonout,752) lat(k),(zntempmax(k)-zntempmin(k))/2., &
          10.8*abs(x(k)),15.5*abs(x(k)),6.2*abs(x(k))
         enddo
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

!  CO2 CONDENSATION
      write(modelout,770)
 770  format(/ 'CO2 CONDENSATION')
     write(modelout,772) ann_co2iceave
     write(modelout,773) ((pco2i - ann_pco2ave)/pco2i)*100.
 772  format(2x,'planet average co2 ice cloud coverage = ',f5.3) 
 773  format(2x,'atmospheric CO2 % loss to surface = ',f6.3) 



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
!--------------------------------------------------------------------------------
	   
	   
	     

!-----------------------------------------------------------------------------------------

       SUBROUTINE WEATHERING2(area, k, pco2, nbelts, focean, fwthr, & 
	   wthrate, warea, wco2,wco2e,v,dt, temp)
	   
	
!  CARBONATE-SILICATE WEATHERING from Berner and Kothavala (2001)
IMPLICIT NONE

real*4 :: fwthr, area, warea, focean,  wthrate,  temp, B, fco2, & 
 pearth, kact, krun, pco2, v, dt, wco2, wco2e, pco2e, fco2e, mass, &
Ptot
DIMENSION :: area(nbelts), warea(nbelts), focean(nbelts), wthrate(nbelts), &
 temp(nbelts+1),pco2(nbelts)
integer ::  nbelts , k

pearth = 0.01 ! abiotic pressure value for Earth if plants were to suddenly disappear (Batalha et al. 2016)
kact = 0.09 ! perK  ! activiation energy
krun = 0.045 ! perK ! soil runoff efficiency

! IS THIS APPLIED AT EVERY LATITUDE OR JUST THE ENTIRE GLOBAL PCO2? :/

B = 0.38 ! This  is beta.. Between 0.3 - 0.5 for silicate rocks, mostly between 0.3 and 0.4(Schwartzman and Volk, 1989; Asolekar et al., 1991). 
!For acidic planets (early Mars?) perhaps beta is as low as 0.25 (Berner, 1992; Haqq-Misra et al. 2016). B = 0.38 (default Earth-like value)

warea(k) = area(k)*(1-focean(k))*fwthr   ! weathering area at each latitude band, with weatherable fraction (fwthr)


pco2e = 30.3*pco2(k)  ! effective CO2 pressure, gives earth weathering rate at T = 288 K and pCO2 = (0.01/3.3e-4 = 30.3)

wco2 = wco2e*warea(k)*(((pco2e/pearth))**B) &
         *(exp(kact*(temp(k) - 288.)))* &
       ((1 + krun*(temp(k) - 288.))**0.65)  ! weathering rate g/yr
	   

	   
	   

       if(temp(k).gt.265.8)then
	   wthrate(k) = wco2/wco2e
       ! Param ends at T = 258. K.. fudging it to get weathering rate ratio values down to 263 K.
	   elseif((temp(k).le.265.8).and.(temp(k).ge.263.))then 
	   wthrate(k) = 0.7*wthrate(k)! no weathering below this temperature (eliminates NANS because no possible intersection between temp and weathering rate curve below 266 K)
       wco2 = wco2e*wthrate(k)
	   elseif(temp(k).lt. 263.)then
	   wco2 = 1.e-35
	   wthrate(k) = 1.e-35
	   endif

END ! ends subrouine  


!------------------------------------------------------------------------------------------------

       SUBROUTINE INTERPOLR(tempp, pco2rr, TL, PL, TR, PR,& 
       PLOGL,  PLOGR, pco2rlog)

! bilinear interpolation that interpolates OLR over surface pressure and temperature
Implicit none

real*4:: tempp, pco2rlog, pco2rr, PLOGR, PLOGL
integer, parameter :: sizep = 16, sizet = 25
integer :: k, P, T, TL, TR, PR, PL
real :: tempgrid(sizet)
real :: pressgrid(sizep)
 
data tempgrid/150., 160., 170., 180., 190., 200., 210., 220., 230., &
     240., 250.,260., 270., 280., 290., 300., 310., 320., 330., 340., & 
	 350., 360., 370., 380., 390./ ! 25 temperatures

data pressgrid/1.e-5, 1.e-4, 1.e-3, 1e-2, 1.e-1, &
      1., 2., 3., 4., 5., 10., 15., &
      20., 25., 30., 35./ 	! 16 CO2 pressures 

		 
		 
	   do P=1,sizep  ! for sizep pressures
		PR=P 
		 

		 
	     if (pco2rr.le.pressgrid(P)) then
		 exit!If an individial p greater than curent p set Ls=9 
		 else
	     PR = sizep + 1
		 endif
	   enddo
	   

       do T = 1,sizet  ! For sizet temperatures
		 TR = T  
		if (tempp.le.tempgrid(T))then 
		exit
        else		
		TR= sizet + 1
        endif		
	      enddo	  
		  
	  
!       For pressures within the grid
	IF(PR.eq.1) THEN
  	PL = 1	! P1 is at or below the lowest grid pressure
	PR = 2
	

	
	PLOGR=log(pressgrid(PR))! upper grid boundary in log units
	PLOGL=log(pressgrid(PL))! lower grid boundary in log units
	pco2rlog=log(pco2rr) ! location of pressure within grid in log units

    
	
       
	ELSEIF(PR.ge.sizep) THEN !	For pressures outside of grid:
 	PR = sizep		! P1 is at or above the highest grid pressure
	PL = sizep - 1
    PLOGR=log(pressgrid(PR))! upper grid boundary in log units
	PLOGL=log(pressgrid(PL))! lower grid boundary in log units
	pco2rlog=log(pco2rr) ! location of pressure within grid in log units
	
	
	
	
        ELSE
	PL = PR-1
	PLOGR=log(pressgrid(PR))! upper grid boundary in log units
	PLOGL=log(pressgrid(PL))! lower grid boundary in log units
	pco2rlog=log(pco2rr) ! location of pressure within grid in log units
	ENDIF
	  

    ! For temperatures within the grid
	if (TR .le. 1) then   ! TL is left-most temperature grid point. TR is right-most temperature grid pointc
	   TL=1
	   TR= 2
          
	elseif (TR .ge.sizet) then
	   TR = sizet
	   TL = sizet - 1
          
	else
          
	   TL = TR-1  
	endif

                                 
       END! ends subroutine

!=====================================================================


       SUBROUTINE INTERPPALB(as,muu, ZYL, ZYR, zyy, AL, AR)

! bilinear interpolation that interpolates OLR over surface pressure and temperature
Implicit none

real*4 :: tempp, pco2rlog, pco2rr, PLOGR, PLOGL, as, muu, ZYY
integer, parameter :: sizep = 16, sizet = 25, sizezy = 19, sizesalb = 21, &
sizemu = sizezy
integer :: k, P, T, TL, TR, PR, PL, A, AL, AR, MU, MUL, MUR, ZY,  ZYL, ZYR
real :: tempgrid(sizet)
real :: pressgrid(sizep)
real :: zygrid(sizezy)
real :: salbgrid(sizesalb)
real :: mugrid(sizezy)
 
data tempgrid/150., 160., 170., 180., 190., 200., 210., 220., 230., &
     240., 250.,260., 270., 280., 290., 300., 310., 320., 330., 340., & 
	 350., 360., 370., 380., 390./ ! 25 temperatures

data pressgrid/1.e-5, 1.e-4, 1.e-3, 1e-2, 1.e-1, &
      1., 2., 3., 4., 5., 10., 15., &
      20., 25., 30., 35./ 	! 16 CO2 pressures 
	  

data zygrid/0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., & 
     55., 60., 65., 70., 75., 80., 85., 90./ ! 19 zenith angle points

data salbgrid/0.0, 0.05, 0.1, 0.15, 0.2,0.25,0.3,0.35,0.4,0.45,  & 
     0.5,0.55, 0.6,0.65,0.7, 0.75,0.8,0.85,0.9,0.95,1./ ! 21 surface albedo points 	  
	  
	  
     dO k = 1, size(mugrid)! 19 cos(zenith angle) points 
	 
	 mugrid(k) = cos(zygrid(k)*3.14159/180.)
	 
	 enddo 
	 
ZYY = acos(muu)*180./3.14159  ! converting mu(k) from code to zenith angle
		 
		 


       do A = 1,sizesalb  ! For sizesalb surface albedo values
		 AR = A  
		if (as.le.salbgrid(A))then 
		exit
        else		
		AR= sizesalb + 1
        endif		
      enddo	  
		  



	       do ZY = 1,sizezy  ! For sizemu zenith angle values.. Here MUL is bigger than MUR
		 ZYR = ZY  
		if (ZYY.le.zygrid(ZY))then 
		exit
        else		
		ZYR= sizezy + 1
        endif		
	      enddo	

	
	! For surface albedo values within the grid
    if (AR .le. 1) then   ! AL is left-most surface albedo grid point. AR is right-most surface albedo grid point
	   AL=1
	   AR= 2
          
	elseif (AR .ge.sizesalb) then
	   AR = sizesalb
	   AL = sizesalb - 1       
	else
	   AL = AR-1  
	endif
	
	
	! For zenith angles (mu) within the grid
   if (ZYR .le. 1) then   ! ZYL is left-most zenith angle grid point. ZYR is right-most zenith angle grid point
	   ZYL=1
	   ZYR= 2
          
	elseif (ZYR .ge.sizezy) then
	   ZYL = sizezy
	   ZYR = sizezy - 1       
	else
	   ZYL = ZYR - 1  
	endif
	
	
                                 
       END! ends subroutine
	   
!================================================================================== 
	   

      SUBROUTINE WRAPUP(last, k,ann_tempave, ann_albave, ann_fluxave, ann_irave, ann_wthrateave, &
	  ann_pco2ave, ann_ph2oave, ann_psurfave, wco2e,d, pco2, nbelts, modelout,zntempmin, tcalc, orbitcount)
	  
	 
	   
	  
	  IMPLICIT NONE
	  integer :: nbelts, last,k, orbitcount, i
	  REAL*4 :: ann_albave, ann_fluxave, ann_irave, ann_wthrateave, wco2e,d, pco2, &
	  zntempmin, ann_tempave, tcalc, fdge, ann_pco2ave, ann_ph2oave, ann_psurfave, &
	  surfh2o

	  DIMENSION :: zntempmin(nbelts), pco2(nbelts)
      integer :: modelout
	  
	  surfh2o =(ann_ph2oave/ann_psurfave)*100.
	  
	  fdge = 1.64
!	  write(40,*)ann_tempave, ann_albave, ann_fluxave, ann_wthrateave, wco2e, ann_pco2ave, surfh2o, ann_psurfave, d, tcalc, orbitcount 
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
!      write(modelout,1016) 'seasonal-average weathering rate = ', &
!      ann_wthrateave*wco2,' g/year'
      write(modelout,1016) 'seasonal-average weathering rate = ', &
      ann_wthrateave*wco2e*fdge,' g/year'
      write(modelout,1016) 'co2 partial pressure = ', ann_pco2ave, ' bars'
 1015 format(3x,a,f7.3,a)
      write(modelout,1017) 'surface water vapor % (by volume) = ', surfh2o
      write(modelout,1016) 'total surface pressure = ', ann_psurfave, ' bars'
 
      write(modelout,1016) 'thermal diffusion coefficient (D) = ', d, & 
      ' Watts/m^2 K'
 1016 format(3x,a,1pe12.5,a)
      write(modelout,1020) 'convergence to final temperature profile in ', &
      tcalc, ' seconds'
 1017 format(3x,a,f7.3,a)
 
 
 1020 format(3x,a,1pe12.5,a)
      write(modelout,1030) 'calculation completed after ', orbitcount+1, ' orbits'
 1030 format(3x,a,i3,a)
 
  
       
      end   ! ENDS SUBROUTINE

!-----------------------------------------------------------------------------



!---------------------------------------------------------------------c
      subroutine  keplereqn(m,e,x)
!---------------------------------------------------------------------c
     
	  integer :: j, n,nc
      real, parameter :: pi=3.141592653589793
      real*8 :: ms, e, x, m, k, del, f, fp, fpp, fppp, dx1, dx2, dx3
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
real*4 :: latrad,  lat, midlatrad,  midlat, x,  coastlat, ocean,focean


dimension :: latrad(nbelts + 1), lat(nbelts + 1), midlatrad(nbelts), &  
       midlat(nbelts), x(nbelts+1), focean(nbelts)


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
      REAL*4 :: a,b,s,EPS,latrad,ob 
      PARAMETER (EPS=1.e-6, JMAX=20) 
      INTEGER :: j 
      REAL*4 olds,sumsteps  ! GOT RID of this weird extra OLDS???
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
      REAL*4 :: a,b,s,insolation1, insolation2, insolation3, latrad,ob
!      EXTERNAL insolation
      INTEGER it,j 
      REAL*4 :: del,sumi,tnm,x 
	  
	 
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
































