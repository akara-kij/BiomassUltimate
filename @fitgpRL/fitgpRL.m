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

classdef fitgpRL < fitsvmRL
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit the linear model %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % List of class attribute
    % List of class attribute
    properties (Access = protected)

    end
    
    methods

    end
      
    properties (Access = public)

    end
    
    % Overried the abstract behavior
    methods( Access = public )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to make prediction through the trained model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ pred,predSD,PredInt ]  = predict(me,dataTest,numMdl)
            if nargin  <= 2
                % Count number of fields in the given structure
                numMdl  = numel( fieldnames(me.mdl) );
            end
            [ pred,predSD,PredInt ] = predict( me.mdl.(me.nameList{numMdl}), dataTest );
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = fitgpRL(cellFeatureName,strResponseVarName)
            me = me@fitsvmRL(cellFeatureName,strResponseVarName);
            me.mdlTitle = "Guassian Process";
        end
    end
    
    methods( Access = protected )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function mdl = fitmodel(me,TBL,varname,varargin)
            mdl = fitrgp(TBL,varname,varargin{:});
        end
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    case  lower('formula')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('Fitmethod')
                        % 'none'    :	No estimation, use the initial parameter values as the known parameter values.
                        % 'exact'	:   Exact Gaussian process regression. Default if n ≤ 2000, where n is the number of observations.
                        % 'sd'      :   Subset of data points approximation. Default if n > 2000, where n is the number of observations.
                        % 'sr'      :   Subset of regressors approximation.
                        % 'fic'     :   Fully independent conditional approximation.
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('BasisFunction')
                        % 'none'
                        % 'constant'
                        % 'linear'
                        % 'Function handle'
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('Beta')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('Sigma')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('SigmaLowerBond')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('Standardize')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('CategoricalPredictors')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('Regularization')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('ComputationalMethod')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('KernelFunction')
                        % 'exponential'             : Exponential kernel.
                        % 'squaredexponential'      : Squared exponential kernel.
                        % 'matern32'                : Matern kernel with parameter 3/2.
                        % 'matern52'                : Matern kernel with parameter 5/2.
                        % 'rationalquadratic'       : Rational quadratic kernel.
                        % 'ardexponential'          : Exponential kernel with a separate length scale per predictor.
                        % 'ardsquaredexponential'   : Squared exponential kernel with a separate length scale per predictor.
                        % 'ardmatern32'             : Matern kernel with parameter 3/2 and a separate length scale per predictor.
                        % 'ardmatern52'             : Matern kernel with parameter 5/2 and a separate length scale per predictor.
                        % 'ardrationalquadratic'	: Rational quadratic kernel with a separate length scale per predictor.
                        % Function handle	A function handle that can be called like this:
                        %        Kmn = kfcn(Xm,Xn,theta)
                        %  where Xm is an m-by-d matrix, Xn is an n-by-d matrix and Kmn is an m-by-n matrix of kernel products such that Kmn(i,j) is the kernel product between Xm(i,:) and Xn(j,:).
                        %  theta is the r-by-1 unconstrained parameter vector for kfcn.
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('KernalParameters')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('DistanceMethod')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('ActiveSet')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('ActiveSetSize')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('ActiveSetMethod')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('RandomSearchSetSize')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('ToleranceActiveSet')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('NumActiveSetRepeats')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('PredictMethod')
                        % 'exact'   :   Exact Gaussian process regression method. Default, if n ≤ 10000.
                        % 'bcd'     :   Block coordinate descent. Default, if n > 10000.
                        % 'sd'      :   Subset of data points approximation.
                        % 'sr'      :   Subset of regressors approximation.
                        % 'fic'     :   Fully independent conditional approximation.
                        me.varList{i  }     = varargin{i  };
                    case  lower('BlockSizeBCD')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};     me.varList{i+1}     = varargin{i+1};     
                    case  lower('NumGreedyBCD')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('Optimizer')
                        % 'quasinewton' : Dense, symmetric rank-1-based, quasi-Newton approximation to the Hessian
                        % 'lbfgs'       : LBFGS-based quasi-Newton approximation to the Hessian
                        % 'fminsearch'	: Unconstrained nonlinear optimization using the simplex search method of Lagarias et al. [5]
                        % 'fminunc'     : Unconstrained nonlinear optimization (requires an Optimization Toolbox™ license)
                        % 'fmincon'     : Constrained nonlinear optimization (requires an Optimization Toolbox license)
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('OptimizerOptions')
                        % 'fminsearch'              : optimset (structure)
                        % 'quasinewton' or 'lbfgs'	: statset('fitrgp') (structure)
                        % 'fminunc' or 'fmincon'	: optimoptions (object)
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};  
                    case  lower('IntialStepSize')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('Verbose')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('OptimizeHyperparameters')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('HyperparameterOptimizationOptions')
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