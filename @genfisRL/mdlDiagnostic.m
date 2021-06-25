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

    %%%%%%%%%%%%%%%%%%%%%
    % Routine Parameter %
    %%%%%%%%%%%%%%%%%%%%%

    % Plot membership functions for input or output variable
    figplanes       = figure;
    numInputs       = length(mdl.Inputs);
    nCase           = nchoosek(numInputs,2);        % Number of outcome
    numOutputs      = length(mdl.Outputs);
    nCk             = nchoosek( 1:numInputs,2);     % Combination of sample space
    nAll            = numInputs+numOutputs*nCase; 
    plotcols        = ceil(sqrt(nAll));
    plotrows        = ceil(nAll/plotcols);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Exhibit the program process %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf("Process of %s : \n", me.mdlTitle);
    strLength = strlength("Process of : ") + strlength(me.mdlTitle);
    fprintf("%s \n",me.underline(strLength));
    
    % Input Membership Function
    for i = 1:numInputs
        subfig      = subplot(plotrows ,plotcols,i,'Parent',figplanes);
        mfType  = 'input';
        mfTitle = 'Input Membership Function ';
        idx     = i;
        plotmf(mdl,mfType,idx);
        xlabel(me.featureName(i),'interpreter','latex');
        title(mfTitle,'interpreter','latex');
        ylabel("MF Parameters",'interpreter','latex');
        xlim([0 100]);
        grid on;
        grid minor;   
        set(subfig,'FontName','Times New Roman');
    end
    
    % Output Membership Function
    for j = 1:numOutputs
        for k = 1:nCase
            subfig      = subplot(plotrows ,plotcols,numInputs+j*k,'Parent',figplanes);
            mfType  = 'output';
            mfTitle = strcat(me.ResponseVarName(j)," Membership Function");
            gensurf(mdl,gensurfOptions('InputIndex',nCk(k,:),'OutputIndex',j));
            title(mfTitle,'interpreter','latex');
            xlim([0 100]);
            xlabel( me.featureName( nCk(k,1) ),'interpreter','latex');
            ylabel( me.featureName( nCk(k,2) ),'interpreter','latex');
            zlabel("MF Parameters",'interpreter','latex');
            grid on;
            grid minor;
            set(subfig,'FontName','Times New Roman');
        end
    end

    % Set the figure position
    set(figplanes, 'Name', 'Membership Function')
    set(figplanes,'units', 'normalized','outerposition',[0. 0. 1 1]);
    
end

