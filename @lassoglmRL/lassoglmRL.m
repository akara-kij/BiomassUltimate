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

classdef lassoglmRL < lassoRL
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit the Lasso or elastic net regularization for %
    % generalized linear models                                  %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Use maximum likelihood
    properties (GetAccess = public, SetAccess = protected)
        Distribution;
        Link;
    end
    
    methods
        function set.Distribution(me,value)
            me.Distribution  = value;
        end
        function set.Link(me,value)
            me.Link  = value;
        end
    end
    
    % Overried the abstract behavior
    methods( Access = public )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to make prediction through the trained model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function pred   = predict(me,dataTest,numMdl)
            if nargin  <= 2
                % Count number of fields in the given structure
                numMdl  = numel( fieldnames(me.mdl) );
            end
            pred        = glmval( me.mdl.(me.nameList{numMdl}), dataTest{:,:},me.Link );
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = lassoglmRL(cellFeatureName,strResponseVarName)
            me = me@lassoRL(cellFeatureName,strResponseVarName);
            me.mdlTitle = "Lasso Generlaized Linear Model";
            % 'normal'
            % 'binomial'
            % 'poisson'
            % 'gamma'
            % 'inverse gaussian'
            me.Distribution  = 'normal';
        end
    end
    
    methods( Access = protected )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function matB = fitmodel(me,TBL,varname,varargin)
            YData               = TBL.(varname);
            TBL.(varname)       = [];
            XData               = TBL{ :,: }; 
            [matB,info]         = lassoglm(XData,YData,me.Distribution,varargin{:});
            matB                = [info.Intercept; matB];
            if isempty( me.mdl )
                numMdl  = 1;
            else
                numMdl          = numel( fieldnames(me.mdl) ) + 1;
            end
            me.mdlInfo.(me.nameList{numMdl}) = info;
        end
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    case  lower('Distribution')
                        % 'normal'
                        % 'binomial'
                        % 'poisson'
                        % 'gamma'
                        % 'inverse gaussian'
                        me.Distribution     = varargin{i+1};
                    case  lower('Alpha')
                        % Alpha = 1 : Lasso Regression
                        % Alpha = 0 : Ridge Regression
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('DFmax')
                        % Maximum number of nonzero coefficients
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Link')
                        % 'comploglog'    : log(–log((1 – µ))) = Xb   
                        % 'identity', default for the distribution 'normal' : µ = Xb
                        % 'log', default for the distribution 'poisson'     : log(µ) = Xb
                        % 'logit', default for the distribution 'binomial'  : log(µ/(1 – µ)) = Xb
                        % 'loglog'        : log(–log(µ)) = Xb
                        % 'probit'	      : Φ–1(µ) = Xb, where Φ is the normal (Gaussian) cumulative distribution function
                        % 'reciprocal', default for the distribution 'gamma': µ–1 = Xb
                        % p (a number), default for the distribution 'inverse gaussian' (with p = –2) : µp = Xb
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};
                        me.Link             = varargin{i+1};
                    case  lower('Intercept')
                        % true of false
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Lambda')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};
                        me.Lambda           = varargin{i+1};
                    case  lower('LambdaRatio')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('MaxIter')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('MCReps')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('NumLambda')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Offset')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('Options')
                        % UseParallel   : Set to true to compute in parallel. The default is false.
                        % UseSubstreams : Set to true to compute in parallel in a reproducible fashion. For reproducibility, set Streams to a type allowing substreams: 'mlfg6331_64' or 'mrg32k3a'. The default is false.
                        % Streams       : A RandStream object or cell array consisting of one such object. If you do not specify Streams, then lassoglm uses the default stream.
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('RelTol')
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('Standardize')
                        % true (default) | false
                        me.varList{i}       = varargin{i};
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('model')
                        me.Model            = lower( varargin{i+1} );
                    case  lower('Lambda')
                        me.Lambda           = varargin{i+1};   
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