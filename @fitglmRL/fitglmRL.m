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

classdef fitglmRL < fitsvmRL
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit the generalized linear model %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % List of class attribute
    properties (Access = protected)
        % List of class attribute
    end
      
    properties (Access = public)
        % 'constant'        : Model contains only a constant (intercept) term.
        % 'linear'          : Model contains an intercept and linear term for each predictor.
        % 'interactions'    : Model contains an intercept, linear term for each predictor, and all products of pairs of distinct predictors (no squared terms).
        % 'purequadratic'   : Model contains an intercept term and linear and squared terms for each predictor.
        % 'quadratic'       : Model contains an intercept term, linear and squared terms for each predictor, and all products of pairs of distinct predictors.
        % 'polyijk'         : Model is a polynomial with all terms up to degree i in the first predictor, degree j in the second predictor, and so on. Specify the maximum degree for each predictor by using numerals 0 though 9. The model contains interaction terms, but the degree of each interaction term does not exceed the maximum value of the specified degrees. For example, 'poly13' has an intercept and x1, x2, x22, x23, x1*x2, and x1*x22 terms, where x1 and x2 are the first and second predictors, respectively.
        Formula;
    end
    
    % Overried the abstract behavior
    methods( Access = public )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to make prediction through the trained model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [pred,predCI]   = predict(me,dataTest,numMdl)
            if nargin  <= 2
                % Count number of fields in the given structure
                numMdl  = numel( fieldnames(me.mdl) );
            end
            [pred,predCI]        = predict( me.mdl.(me.nameList{numMdl}), dataTest );
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = fitglmRL(cellFeatureName,strResponseVarName)
            me = me@fitsvmRL(cellFeatureName,strResponseVarName);
            me.mdlTitle = "Generalized Linear Model";
            me.Formula  = 'Linear';
        end
        
    end
    
    methods( Access = protected )

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function mdl = fitmodel(me,TBL,varname,varargin)
            mdl = fitglm(TBL,me.Formula,varargin{:});
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    case  lower('BinomialSize')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('B0')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('CategoricalVars')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('DispersionFlag')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Distribution')
                        % 'normal'           : Normal distribution
                        % 'binomial'         : Binomial distribution
                        % 'poisson'          : Poisson distribution
                        % 'gamma'            : Gamma distribution
                        % 'inverse gaussian' : Inverse Gaussian distribution
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Exclude')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Intercept')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Link')
                        % 'identity'        : f(μ) = μ                  μ = Xb
                        % 'log'             : f(μ) = log(μ)             μ = exp(Xb)
                        % 'logit'           : f(μ) = log(μ/(1–μ))       μ = exp(Xb) / (1 + exp(Xb))
                        % 'probit'          : f(μ) = Φ–1(μ), where Φ is the cumulative distribution function of the standard normal distribution.	μ = Φ(Xb)
                        % 'comploglog'      : f(μ) = log(–log(1 – μ))	μ = 1 – exp(–exp(Xb))
                        % 'reciprocal'      : f(μ) = 1/μ                μ = 1/(Xb)
                        % p (a number)      : f(μ) = μp                 μ = Xb1/p
                        % S (a structure)   : f(μ) = S.Link(μ)          μ = S.Inverse(Xb)
                        % with three fields. Each field holds a function handle that accepts a vector of inputs and returns a vector of the same size:
                        %            S.Link — The link function
                        %            S.Inverse — The inverse link function
                        %            S.Derivative — The derivative of the link function
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Options')
                        % Display	: Amount of information displayed by the algorithm
                        %    'off' — Displays no information
                        %    'final' — Displays the final output
                        % MaxIter   : Maximum number of iterations allowed, specified as a positive integer
                        % TolX      : Termination tolerance for the parameters, specified as a positive scalar
                        % Example   : 'Options',statset('Display','final','MaxIter',1000) 
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Offset')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('ResponseVar')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
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