%%% 
%%% params.m
%%% 
%%% Automatically-generated Matlab file containing parameters
%%% used in this MITgcm experiment.
%%% 

vectorInvariantMomentum=1;
viscAr=1.00000000e-04;
viscA4=0.00000000e+00;
viscAh=0.00000000e+00;
viscA4Grid=0.00000000e+00;
viscAhGrid=0.00000000e+00;
viscA4GridMax=5.00000000e-01;
viscAhGridMax=1.00000000e+00;
useAreaViscLength=0;
useFullLeith=1;
viscC4leith=0.00000000e+00;
viscC4leithD=0.00000000e+00;
viscC2leith=0.00000000e+00;
viscC2leithD=0.00000000e+00;
viscC2smag=0.00000000e+00;
viscC4smag=4.00000000e+00;
diffKrT=0.00000000e+00;
diffKhT=0.00000000e+00;
tempAdvScheme=33;
saltAdvScheme=33;
multiDimAdvection=1;
tempStepping=1;
saltStepping=1;
staggerTimeStep=1;
eosType='JMD95Z';
no_slip_sides=0;
no_slip_bottom=0;
bottomDragLinear=0.00000000e+00;
bottomDragQuadratic=0.00000000e+00;
f0=0.00000000e+00;
gravity=9.81000000e+00;
rhonil=1.00000000e+03;
rhoConst=1.00000000e+03;
ivdc_kappa=0.00000000e+00;
implicitDiffusion=1;
implicitViscosity=1;
nonHydrostatic=0;
exactConserv=1;
useCDscheme=0;
readBinaryPrec=64;
useSingleCpuIO=1;
debugLevel=-1;
vectorInvariantMomentum=1;
useJamartWetPoints=1;
useJamartMomAdv=1;
diffK4T=3.90625000e+02;
diffK4S=3.90625000e+02;

useSRCGSolver=1;
cg2dMaxIters=1000;
cg2dTargetResidual=1.00000000e-12;
cg3dMaxIters=300;
cg3dTargetResidual=1.00000000e-07;

momDissip_In_AB=0;
tracForcingOutAB=1;
nIter0=0;
abEps=1.00000000e-01;
chkptFreq=4.32000000e+03;
pChkptFreq=4.32000000e+03;
taveFreq=0.00000000e+00;
dumpFreq=0.00000000e+00;
monitorFreq=8.64000000e+04;
cAdjFreq=0.00000000e+00;
dumpInitAndLast=1;
pickupStrictlyMatch=0;
endTime=86400;
deltaT=5.00000000e+01;

usingCartesianGrid=1;
usingSphericalPolarGrid=0;
delX=[ 5.00000000e+01 ];
delY=[ 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 5.00000000e+01 ];
delR=[ 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 2.00000000e+00 ];

bathyFile='bathyFile.bin';
hydrogThetaFile='hydrogThetaFile.bin';
hydrogSaltFile='hydrogSaltFile.bin';

useDiagnostics=1;
useKPP=1;
useRBCS=1;
useSEAICE=0;
useEXF=0;

diag_fields{1,1}='ETAN';
diag_fileNames{1}='ETAN_inst';
diag_frequency(1)=-4.32000000e+03;
diag_timePhase(1)=0.00000000e+00;
diag_fields{1,2}='UVEL';
diag_fileNames{2}='UVEL_inst';
diag_frequency(2)=-4.32000000e+03;
diag_timePhase(2)=0.00000000e+00;
diag_fields{1,3}='VVEL';
diag_fileNames{3}='VVEL_inst';
diag_frequency(3)=-4.32000000e+03;
diag_timePhase(3)=0.00000000e+00;
diag_fields{1,4}='WVEL';
diag_fileNames{4}='WVEL_inst';
diag_frequency(4)=-4.32000000e+03;
diag_timePhase(4)=0.00000000e+00;
diag_fields{1,5}='THETA';
diag_fileNames{5}='THETA_inst';
diag_frequency(5)=-4.32000000e+03;
diag_timePhase(5)=0.00000000e+00;
diag_fields{1,6}='SALT';
diag_fileNames{6}='SALT_inst';
diag_frequency(6)=-4.32000000e+03;
diag_timePhase(6)=0.00000000e+00;

useRBCtemp=1;
useRBCsalt=1;
useRBCuVel=0;
useRBCvVel=0;
tauRelaxT=4.32000000e+03;
tauRelaxS=4.32000000e+03;
tauRelaxU=4.32000000e+03;
tauRelaxV=4.32000000e+03;
relaxTFile='sponge_temp.bin';
relaxSFile='sponge_salt.bin';
relaxMaskFile{1}='rbcs_temp_mask.bin';
relaxMaskFile{2}='rbcs_salt_mask.bin';


