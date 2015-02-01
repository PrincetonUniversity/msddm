% Compute Chi sq for one set of (correct or incorrect) RTs
% QUANTILE WEIGHTS HERE
function [val, df] = chisq(rtData,tArray,ddmcdf,nTotalTrials)
nTrials = length(rtData);

qBins = [.1 .2 .2 .2 .2 .1];
cpv = cumsum(qBins);
q = quantile(rtData,cpv);
q(end) = inf; % Matlab is lame
qcData = nTrials*qBins;

% checking work...
% qcData = histc(rtData,[-inf q]);
% qcData(end) = [];

%% Expected # of trials in each quantile
[cPs] = interp1(tArray,ddmcdf,q);
qcExp = diff([0 cPs]) * nTotalTrials;
qcExp = max(qcExp, 1e-10*ones(1,length(qBins)));


df = (qcData-qcExp)/sqrt(qcExp);

val = norm(df).^2;