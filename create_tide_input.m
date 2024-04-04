% Generate example tidal input files.

% Tidal input files are real*4 IEEE big-endian binary
% with dimenstion OBlength * tidalComponents,
% where OBlength is the length of the open boundary
% and tidalComponents is the number of tidal components
% specified in OBCS_PARAMS.h.

% OB[N,S,E,W][am,ph]File :: Files with boundary conditions,
%                           the letter combinations mean:
%              N/S/E/W   :: northern/southern/eastern/western boundary
%              am/ph     :: tidal amplitude (m/s) / phase (s)

% Tidal periods are specified using variable tidalPeriod in data.obcs
% Tidal amplitude is the maximum tidal velocity in m/s.
% Tidal phase indicates time in s of maximum positive tide relative
% to model startTime=0.

% readbin.m and writebin.m are in MITgcm-master/utils/matlab/cs_grid/read_cs

% create tidal input files
nx=Nx; ny=Ny;
tidalComponents=1;

amp_tide=0.1; %velocity magnitude of tidal amplitude (m/s)

for ob={'N','S'}%,'E','W'}
    OBlength=ny;
    if any(strcmp(ob,{'N','S'}))
        OBlength=nx;
    end
    for fld={'am','ph'}
        fnm=['OB' ob{1} fld{1} 'File'];
        tmp=randn(OBlength,tidalComponents)/1000;

        % specify (0.1 m/s, 0 hr) for North boundary tidal component 1
        if strcmp(ob,'N')
            if strcmp(fld,'am')
                tmp(:,1) = tmp(:,1) + amp_tide;
            else
                tmp(:,1) = tmp(:,1) + 0 * 3600;
            end
        end
size(tmp)
        % specify (0.1 m/s, 0 hr) for South boundary tidal component 2
        if strcmp(ob,'S')
            if strcmp(fld,'am')
                tmp(:,1) = tmp(:,1) + amp_tide;
            else
                tmp(:,1) = tmp(:,1) + 0 * 3600;
            end
        end
tmp
fnm
size(tmp)
        %writebin(fnm,tmp)
        writeDataset(tmp,fullfile(inputpath,strcat(fnm,'.bin')),ieee,prec);
    end
end

