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

classdef fitsvmRL < RegLearner
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit the support vector machine %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % List of class attribute
    properties (Access = protected)
        % List of class attribute
    end
      
    properties (Access = public)
        % The anchor state of the random generation 
        sRand = RandStream('mcg16807','Seed',144);
    end
    
    % Overried the abstract behavior
    methods( Access = public )
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to fit the linear model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        me = fit(me,varargin)
        mdlDiagnostic(me,mdl)
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to make prediction through the trained model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function pred   = predict(me,dataTest,numMdl)
            if nargin  <= 2
                % Count number of fields in the given structure
                numMdl  = numel( fieldnames(me.mdl) );
            end
            pred        = predict( me.mdl.(me.nameList{numMdl}), dataTest );
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = fitsvmRL(cellFeatureName,strResponseVarName)
            me = me@RegLearner(cellFeatureName,strResponseVarName);
            me.mdlTitle = "Support Vector Machine";
        end
    end
    
    methods( Access = protected )

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function mdl = fitmodel(me,TBL,varname,varargin)
            mdl = fitrsvm(TBL,varname,varargin{:});
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    case  lower('Standardize')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('KernelFunction')
                        % 'gaussian','rbf' : Gaussian or Radial Basis Function (RBF) kernel
                        % 'linear'         : Linear kernel
                        % 'polynomial'     : Polynomial kernel. Use 'PolynomialOrder',q to specify a polynomial kernel of order q
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('BoxConstraint')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('KernelScale')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('PolynomialOrder')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Epsilon')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('KernelOffeset')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Solver')
                        % 'ISDA'    : Iterative Single Data Algorithm 
                        % 'L1QP'	: Uses quadprog (Optimization Toolbox) to implement L1 soft-margin minimization by quadratic programming.
                        % 'SMO'     : Sequential Minimal Optimization 
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Alpha')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('CacheSize')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('ClipAlphas')
                        % true  : At each iteration, if αj is near 0 or near Cj, then MATLAB sets αj to 0 or to Cj, respectively.
                        % false	: MATLAB does not change the alpha coefficients during optimization.
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('OptimizeHyperparameters')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('HyperparameterOptimizationOptions')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('RemoveDuplicates')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('NumPrint')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('OutlierFraction')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('RemoveDuplicates')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};                        
                    case  lower('Verbose')
                        % 0 : The software does not display or save convergence information.
                        % 1 : The software displays diagnostic messages and saves convergence criteria every numprint iterations, where numprint is the value of the name-value pair argument 'NumPrint'.
                        % 2	: The software displays diagnostic messages and saves convergence criteria at every iteration.
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