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

classdef ridgeRL < fitsvmRL
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit the Ridge Regularized Linear Regression %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Default is standardzie N(0,1) by set Scaled = 1
    %    Therefore, coefficients are displayed on the same scale.
    % List of class attribute
    properties (Access = protected)
        Model;
        DesignMat;
    end
    properties (SetAccess = protected, GetAccess = public)
        Lambda;
        Scaled;
    end
    methods
        function set.Model(me,value)
            me.Model        = value;
        end
        function set.DesignMat(me,value)
            me.DesignMat    = value;
        end
        function set.Lambda(me,value)
            me.Lambda       = value;
        end
        function set.Scaled(me,value)
            me.Scaled       = value;
        end
    end
      
    properties (Access = public)

    end
    
    % Overried the abstract behavior
    methods( Access = public )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to fit the ridge regression %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        me = fit(me,varargin)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to make prediction through the trained model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function pred  = predict(me,dataTest,numMdl)
            if nargin  <= 2
                % Count number of fields in the given structure
                numMdl  = numel( fieldnames(me.mdl) );
            end
            matD        = x2fx( dataTest{:,:}, me.Model );
            matD(:,1)   = []; % Exclude constant column
            vecB        = me.mdl.(me.nameList{numMdl});
            pred        = vecB(1)+matD*vecB(2:end);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = ridgeRL(cellFeatureName,strResponseVarName)
            me = me@fitsvmRL(cellFeatureName,strResponseVarName);
            me.mdlTitle = "Ridge : Regularized Linear Regression";
            
            % 'linear'          : Constant and linear terms. This is the default.
            % 'interaction'     : Constant, linear, and interaction terms
            % 'quadratic'       : Constant, linear, interaction, and squared terms
            % 'purequadratic'   : Constant, linear, and squared terms
            me.Model    = [];
            me.Lambda   = 0:100; % glmval has default = 100;
            me.Scaled   = 0;
        end
    end
    
    methods( Access = protected )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function matB = fitmodel(me,TBL,varname,varargin)
            idxTBL              = TBL.Properties.VariableNames ~= me.ResponseVarName ;
            XData               = TBL{ :,idxTBL};
            YData               = TBL{ :,varname };
            me.DesignMat        = x2fx(XData,me.Model);
            me.DesignMat(:,1)   = []; % Exclude constant column
            matB                = ridge(YData,me.DesignMat,me.Lambda,me.Scaled);
        end
        
        function [mdl,idx,matMSE]  = optmodel(me,TBL,varname,matB)
            % Coefficient at the min MSE
            idxTBL              = TBL.Properties.VariableNames ~= me.ResponseVarName ;
            XData               = TBL{ :,idxTBL  };
            YData               = TBL{ :,varname };
            me.DesignMat        = x2fx(XData,me.Model); 
            me.DesignMat(:,1)   = []; % Exclude constant column
            YMat                = matB(1,:)+me.DesignMat*matB(2:end,:);
            matERR              = YData - YMat;
            matMSE              = mean( matERR.*matERR );
            [~,idx]             = min ( matMSE );
            mdl                 = matB( :, idx );
        end
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    case  lower('model')
                        me.Model            = lower( varargin{i+1} );
                    case  lower('Lambda')
                        me.Lambda           = varargin{i+1};   
                    case  lower('Scaled')
                        %me.Scaled           = varargin{i+1}; 
                    case  lower('kFold')
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