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

classdef fitrlinearRL < lassoRL
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit the high-dimensional linear regression model %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Overried the abstract behavior
    methods( Access = public )
        
        % Routine to make prediction through the trained model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function pred  = predict(me,dataTest,numMdl)
            if nargin  <= 2
                % Count number of fields in the given structure
                numMdl  = numel( fieldnames(me.mdlInfo) );
            end
            pred     = predict(me.mdl.(me.nameList{numMdl}),dataTest);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = fitrlinearRL(cellFeatureName,strResponseVarName)
            me = me@lassoRL(cellFeatureName,strResponseVarName);
            me.mdlTitle = "higher dimensional linear regression";
        end
    end
    
     methods( Access = protected )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function mdl = fitmodel(me,TBL,varname,varargin)
            idx    = strcmp(varargin,'OptimizeHyperparameters');
            strVar = convertStringsToChars(varname);
            if ~isempty( me.Model )
                strVar = me.Model;
            end
            if isempty( me.mdl )
                numMdl  = 1;
            else
                numMdl  = numel( fieldnames(me.mdl) ) + 1;
            end
            if nnz(idx)== 1
                [mdl,info,opt]     = fitrlinear(TBL,strVar,varargin{:});
                 me.optInfo.(me.nameList{numMdl}) = opt;
            else
                [mdl,info]         = fitrlinear(TBL,strVar,varargin{:});
            end
            me.mdlInfo.(me.nameList{numMdl})      = info;
        end
                
        function [mdl,idx,matMSE]  = optmodel(me,TBL,varname,mdlAll)
            % Coefficient at the min MSE
            idxTBL      = TBL.Properties.VariableNames ~= me.ResponseVarName ;
            YData       = TBL{ :,varname };
            YMat        = predict(mdlAll,TBL(:,idxTBL));
            matERR      = YData - YMat;
            matMSE      = mean( matERR.*matERR );
            [~,idx]     = min ( matMSE );
            mdl         = selectModels(mdlAll,idx);
        end
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    case  lower('Lambda')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};
                        me.Lambda           = varargin{i+1};  
                    case  lower('Standardize')
                        if varargin{i+1}   == true
                            me.dataTBL      = normalize(me.dataTBL,'center','mean','scale','std');
                        end
                    case  lower('Epsilon')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('Learner')
                        % 'leastsquares'
                        % 'svm'
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('ObservationsIn')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('Regularization')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};      
                    case  lower('Regularization')
                        % 'lasso'
                        % 'ridge'
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('Solver')
                        % 'sgd'     : Stochastic gradient descent (SGD) [5][3]	 
                        % 'asgd'	: Average stochastic gradient descent (ASGD) [8]	 
                        % 'dual'	: Dual SGD for SVM [2][7]	Regularization must be 'ridge' and Learner must be 'svm'.
                        % 'bfgs'	: Broyden-Fletcher-Goldfarb-Shanno quasi-Newton algorithm (BFGS) [4]	Inefficient if X is very high-dimensional.
                        % 'lbfgs'	: Limited-memory BFGS (LBFGS) [4]	Regularization must be 'ridge'.
                        % 'sparsa'	: Sparse Reconstruction by Separable Approximation (SpaRSA) [6]
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('Beta')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('Bias')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('FitBias')
                        % true   :	The software includes the bias term b in the linear model, and then estimates it.
                        % false	 :  The software sets b = 0 during estimation.    
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('PostFitBias')
                        % true/false
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};       
                    case  lower('Verbose')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('BatchSize')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('LearnRate')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('OptimizeLearnRate')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('TruncationPeriod')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('CategoricalPredictors')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('PredictorNames')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};      
                    case  lower('ResponseName')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('ResponseTransform')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('Weights')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('BatchLimit')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('BetaTolerance')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('NumCheckConvergence')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('PassLimit')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('ValidationData')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('GradientTolerance')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};       
                    case  lower('IterationLimit')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};       
                    case  lower('BetaTolerance')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('GradientTolerance')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('OptimizeHyperparameters')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('HyperparameterOptimizationOptions')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('Formula')
                        me.Model            = lower( varargin{i+1} );
                    case  lower('kFold') % That is 'MCReps' Name-value pair
                        me.KFolds           = varargin{i+1};
                        me.c = cvpartition(nAll,'KFold',Kfolds);
                    case  lower('CrossVal')
                        me.KFolds           = varargin{i+1};
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Holdout')
                        % dividetrain : Assign all targets to training set
                        me.KFolds           = 1;
                        me.chkCV            = false;
                        me.pFactor          = varargin{i+1}; % Array of { pTestData,pValidateData }, pFactor = [0.15,0.15];  
                    case  lower('Leaveout')
                        if varargin{i+1} == true
                            me.c            = cvpartition(nAll,'LeaveOut');
                        end
                    case  lower('Resubstitution')
                        if varargin{i+1} == true
                            % dividetrain : Assign all targets to training set
                            me.KFolds       = 1;
                        end
                    case  lower('cvpartition')
                        me.c                = varargin{i+1};   
                        me.KFolds           = me.c.NumTestSets;
                    otherwise
                end

            end
            
            % Filter out the empty cell    
            me.varList = me.varList(~cellfun('isempty',me.varList)); % Call Built-in string
                
            % Check about the existanc of CVPartition object
            if ~isempty(me.c)
                me.chkCV    = true;
                me.flagShow = false;
            end

            % Create the empty structure of the summary table 
            me.nameList = cell(me.KFolds+1,1);
            for fold = 1:me.KFolds
               me.nameList{fold}       = "fold"+num2str(fold);
            end
            me.nameList{end}           = "CrossVal";
            me.R2TBL                   = [];
            me.EvaTBL                  = [];
            me.Residuals               = [];
            me.mdl                     = [];

        end
    end
    
end