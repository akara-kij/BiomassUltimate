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

function mdlDiagnostic(me,mdl)

    % Routine Parameter
    NComp     =   1:length(mdl.PCTVAR);           % Number of component
    numMdl    =   numel( fieldnames(me.mdl) );    % The latest K-fold
    mdl       =   me.mdl.( me.nameList{numMdl} ); % The latest model
    MSize     =   64;
    MColor    =   [0.00,0.45,0.74];
    idxTBL    =   me.dataTBL.Properties.VariableNames ~= me.ResponseVarName ;
    feature   =   categorical( me.dataTBL.Properties.VariableNames(idxTBL) );
    
    % Compute the V.I.P. score
    W0        = mdl.stats.W ./ sqrt(sum(mdl.stats.W.^2,1));
    N         = size(mdl.XLoad,1);
    sumSq     = sum(mdl.XScore.^2,1).*sum(mdl.YLoad.^2,1);
    vipScore  = sqrt(N* sum(sumSq.*(W0.^2),2) ./ sum(sumSq,2));
    idxVIP    = find(vipScore >= 1);

    % Inform about the diagnostic process
    fprintf("Process of %s : \n", me.mdlTitle);
    strLength = strlength("Process of : ") + strlength(me.mdlTitle);
    fprintf("%s \n",me.underline(strLength));

    strTitle  = "Partial Least-Square Regression";
    figure('name',strTitle);
    
    subplot(2,1,1)
    strTitle = "Component Selection";
    plot(NComp,cumsum(100*mdl.PCTVAR(2,:)),'-o','Color',MColor,'MarkerFaceColor',MColor);
    title(strTitle,'interpreter','latex');
    xlabel('Number of PLS components','interpreter','latex');
    ylabel('Variance Explained, \%','interpreter','latex');
    axis tight;
    grid on;
    grid minor;
    
    subplot(2,1,2)
    strTitle = "Feature Selection";
    scatter(feature,vipScore,MSize,'o','filled')
    hold on
    scatter(feature(idxVIP),vipScore(idxVIP),MSize,'ro','filled')
    plot([1 length(vipScore)],[1 1],'--k')
    title(strTitle,'interpreter','latex');
    hold off
    axis tight
    xlabel('Predictor Variables','interpreter','latex')
    ylabel('VIP Scores','interpreter','latex')
    axis tight;
    grid on;
    grid minor;
    
end

