%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% # Released under MIT License %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2021 Akara Kijkarncharoensin, akara_kij@utcc.ac.th        %
% Department of Computer Engineering and Financial Technology,            %
% School of Engineering, University of the Thai Chamber of Commerce.      %
%                                                                         %
% Permission is hereby granted, free of charge, to any person obtaining a %
% copy of this software and associated documentation files (the           %
% "Software") , to deal in the Software without restriction, including    %
% without limitation the rights to use, copy, modify, merge, publish,     %
% distribute, sublicense, and/or sell copies of the Software , and to     %
% permit persons to whom the Software is furnished to do so, subject to   %
% the following conditions:                                               %
%                                                                         %
% The above copyright notice and this permission notice shall be included %
% in all copies or substantial portions of the Software.                  %
%                                                                         %
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,         %
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF      %
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  %
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY    %
% CLAIM,DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT% 
% OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR%
% THE USE OR OTHER DEALINGS IN THE SOFTWARE.                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function HypTBL = resHypothesis( me, res, alpha )

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Residual Diagnostic : The Normaility Check %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Parameter of the hypothesis testing
    resHYP = struct( 'KS',[]   ,'LT',[]   ,'JB',[]   ,'AD',[]   );
    idx    = struct( 'KS',true ,'LT',true ,'JB',true ,'AD',true );  
    if nargin == 2 
       alpha = 0.05;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Residual Normalization %
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    resSTD = normalize( res ,'zscore');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Kolmogorov-Smirnov Test %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % H0 : Data comes from a standard normal distribution
    % Ha : It does not come from such a distribution
    try
        % Prevent error from insufficiency about number of data
        resHYP.KS = struct( 'h',NaN,'p',NaN,'stat',NaN,'cv',NaN,'name', 'Kolmogorov-Smirnov' );
        [ resHYP.KS.h, resHYP.KS.p, resHYP.KS.stat, resHYP.KS.cv ] = kstest(resSTD, 'Alpha', alpha );
    catch

    end
    %%%%%%%%%%%%%%%%%%%
    % Lilliefors Test %
    %%%%%%%%%%%%%%%%%%%
    % H0 : Data comes from a distribution in the normal family.
    % Ha : It does not come from such a distribution
    try
        % Prevent error from insufficiency about number of data
        resHYP.LT = struct( 'h',NaN,'p',NaN,'stat',NaN,'cv',NaN,'name', 'Lilliefors' );
        [ resHYP.LT.h, resHYP.LT.p, resHYP.LT.stat, resHYP.LT.cv ] = lillietest(resSTD, 'Distribution', 'normal', 'Alpha', alpha );
    catch

    end

    %%%%%%%%%%%%%%%%%%%%
    % Jarque-Bera Test %
    %%%%%%%%%%%%%%%%%%%%
    % H0 : Data comes from a normal distribution with an unknown mean and variance
    % Ha : It does not come from such a distribution
    try
        % Prevent error from insufficiency about number of data
        resHYP.JB = struct( 'h',NaN,'p',NaN,'stat',NaN,'cv',NaN,'name', 'Jarque-Bera' );
        [ resHYP.JB.h, resHYP.JB.p, resHYP.JB.stat, resHYP.JB.cv ] = jbtest(resSTD, alpha );
    catch

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Anderson-Darling Test %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % H0 : Data is from a population with a normal distribution
    % Ha : It is not from a population with a normal distribution
    try
        % Prevent error from insufficiency about number of data
        resHYP.AD = struct( 'h',NaN,'p',NaN,'stat',NaN,'cv',NaN,'name', 'Anderson-Darling' );
        [ resHYP.AD.h, resHYP.AD.p, resHYP.AD.stat, resHYP.AD.cv ] = adtest(resSTD, 'Distribution', 'norm', 'Alpha', alpha );
    catch

    end
    
    % Summary the hypothesis testing
    varNames    = {'name','h','p','stat','cv'};
    rowNames    = {'KS','LT','JB','AD'};
    nameTBL     = {resHYP.KS.name,resHYP.LT.name,resHYP.JB.name,resHYP.AD.name}';
    hTBL        = [resHYP.KS.h   ,resHYP.LT.h   ,resHYP.JB.h   ,resHYP.AD.h   ]';
    pTBL        = [resHYP.KS.p   ,resHYP.LT.p   ,resHYP.JB.p   ,resHYP.AD.p   ]';
    sTBL        = [resHYP.KS.stat,resHYP.LT.stat,resHYP.JB.stat,resHYP.AD.stat]';
    cvTBL       = [resHYP.KS.cv  ,resHYP.LT.cv  ,resHYP.JB.cv  ,resHYP.AD.cv  ]';
    HypTBL      = table(nameTBL, hTBL, pTBL, sTBL, cvTBL,'VariableNames',varNames,'RowNames',rowNames );

end