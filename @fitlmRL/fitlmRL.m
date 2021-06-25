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

classdef fitlmRL < RegLearner
    
    % Not included standardize feature
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit the linear model %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
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
        me = fit(me,varargin);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to make prediction through the trained model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [pred,predCI] = predict(me,dataTest,numMdl)
            pred   = struct('Step',[],'Robust',[]);
            predCI = pred;
            if nargin  <= 2
                % Count number of fields in the given structure
                numMdl  = numel( fieldnames(me.mdl.Step) );
            end
            [pred.Step  ,predCI.Step  ] = predict( me.mdl.Step.  (me.nameList{numMdl}), dataTest );
            [pred.Robust,predCI.Robust] = predict( me.mdl.Robust.(me.nameList{numMdl}), dataTest );
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to exhibit the list of list formular %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function TBL = showFormula(me)
            strFormula = strings(me.KFolds,1);
            rowNames   = strFormula;
            for fold = 1:me.KFolds
                % Recall the formula of the stepwise lienar model
                strFormula(fold,1) = me.mdl.Step.(me.nameList{fold}).Formula.LinearPredictor; 
                rowNames  (fold,1) = me.nameList{fold};
            end
            % Create the formula table
            varNames = "Stepwise Linear Model";
            TBL      = table( strFormula, 'VariableNames', varNames, 'RowNames', rowNames );
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = fitlmRL(cellFeatureName,strResponseVarName)
            me = me@RegLearner(cellFeatureName,strResponseVarName);
            me.mdlTitle = "Linear Model";
        end
    end
    
    methods( Access = protected )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Setting the parameter of the neural network %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    case  lower('CategoricalVars')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Criterion')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Exclude')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Intercept')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('Lower')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('NSteps')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('PredictorVars')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('PRemove')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('VarNames')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('Weights')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('Standardize')
                        if varargin{i+1}   == true
                            me.dataTBL      = normalize(me.dataTBL,'center','mean','scale','std');
                        end
                    case  lower('PEnter')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                     case  lower('NSteps')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('Verbose')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('upper')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('kFold')
                        me.KFolds           = varargin{i+1};
                        me.c = cvpartition(nAll,'KFold',Kfolds);
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
            me.R2TBL.Step              = [];
            me.R2TBL.Robust            = [];
            me.EvaTBL.Step             = [];
            me.EvaTBL.Robust           = [];
            me.Residuals.Step          = [];
            me.Residuals.Robust        = [];
            me.mdl                     = [];

        end
    end
end