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

classdef rootLearner < handle
    
    % List of public attribute 
    properties( Access = public )   
        % List of class attribute
        featureName     = {};           % List of feature name
        ResponseVarName = '';           % Name of response variable
        dataTBL         = [];           % Table of data
        dataID          = [];           % I.D. of the selected row
        mdl             = [];           % Collection of the class model
    end
    
    % List of public behavior
    methods( Access = public )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = rootLearner(cellFeatureName,strResponseVarName)
            me.featureName     = cellFeatureName;
            me.ResponseVarName = strResponseVarName;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Create strings use for the underline %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function str = underline(me,strLength)
            % Memory allocation for n lines
            str = repmat( '=',1,strLength );
        end
        
        %%%%%%%%%%%%%%%%%
        % Load database %
        %%%%%%%%%%%%%%%%%
        function me = load(me,fileName,ShtName)
            % Download the raw data
            rawData = readtable(fileName,"FileType","spreadsheet","Sheet",ShtName);

            % Extract feature of proximate data analysis
            me.dataTBL = me.extractData(rawData);
         end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Extract the table of raw data %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function dataTBL = extractData( me, origTBL )
            dataTBL = origTBL(:,me.featureName);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Exploratory Data Analysis %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function figData = eda(me)

            % Routine Paramater
            varNames = me.dataTBL.Properties.VariableNames;
            nFeature = length(varNames);

            % Create the new figure
            figData = figure;

            % Matrix Plot
            subplot(2,1,1)
            %[S,AX,BigAx,H,HAx] = plotmatrix([dataTBL.FC dataTBL.VM dataTBL.A dataTBL.HHV]);
            [S,AX,BigAx,H,HAx] = plotmatrix(me.dataTBL{:,:});
            title(BigAx,'Histogram and Correlation','interpreter','latex');

            % Set the limit of x-axis and y-axis
                for ii=1:size(AX,1)
                    for jj = 1:size(AX,2)
                        ylim( AX(ii,jj),[1 100]);
                        xlim( AX(ii,jj),[1 100]);
                        grid(AX(ii,jj),'on')
                        grid(AX(ii,jj),'minor');
                        axis tight;
                    end
                end

                % Set the properties of the histograms
                for ii=1:size(H,2)
                    H(ii).NumBins = 15;
                    H(ii).Normalization = 'pdf';
                    %xlim(HAx(ii),[min(dataTBL{:,ii}) max(dataTBL{:,ii})]);
                    xlim(HAx(ii),[1,100]);
                end

            % Set the label along X and Y axis
            for ii=1:nFeature
                strName = varNames{ii};
                ylabel(AX(ii ,1) ,strName ,'Rotation',90,'HorizontalAlignment','right','interpreter','latex');
                xlabel(AX(end,ii),strName ,'Rotation',0 ,'HorizontalAlignment','right','interpreter','latex');
            end

            % Boxplot
            subplot(2,1,2)
            boxplot(me.dataTBL{:,:},'Labels',varNames);
            title('Discriptive Statistic','interpreter','latex')
            xlabel('Proximate Analysis','interpreter','latex')
            ylim([0 100]);
            axis tight;
            grid on
            grid minor;

            % Set the property of the figure
            strName = "Exploratory Data Analysis";
            set(figData,'name',strName);
            set(figData,'units','normalized','outerposition',[0. 0. 1 1]);

        end
        
    end

end