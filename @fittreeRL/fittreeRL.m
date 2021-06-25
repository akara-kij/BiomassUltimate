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

classdef fittreeRL < fitsvmRL
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit the tree model %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % List of class attribute
    properties (Access = protected)
        % List of class attribute

    end
    
    methods

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
        function me = fittreeRL(cellFeatureName,strResponseVarName)
            me = me@fitsvmRL(cellFeatureName,strResponseVarName);
            me.mdlTitle = "Regression Trees";
        end
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % View the regression tree  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function view(me,varargin)
             
            % Show all trees
            if nargin == 1
                for i=1:me.KFolds
                    view(me.mdl.( me.nameList{i} ),'Mode','graph');
                end
                return
            end
            
            % Name-Value pair option
            for i = 1:2:length(varargin)
                switch lower(varargin{i})
                    case lower('Top')   % View only the first ith top
                        for j=1:varargin{i+1}
                            view(me.mdl.( me.nameList{j} ),'Mode','graph');
                        end
                    case lower('Bottom')% View only the first ith bottom
                        for j=varargin{i+1}:-1:1
                            view(me.mdl.( me.nameList{j} ),'Mode','graph');
                        end
                end
            end
        end
    end
    
    methods( Access = protected )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function mdl = fitmodel(me,TBL,varname,varargin)
            mdl = fitrtree(TBL,varname,varargin{:});
            me.mdlTitle = "Regression tree";
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    %%%%%%%%%%%%%%%%%%%
                    % Hyperparameters %
                    %%%%%%%%%%%%%%%%%%%
                    case  lower('MaxNumSplites')
                            % 'MaxNumSplits' : Maximal number of decision splits
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1}; 
                    case  lower('MinLeafSize')
                            % Minimum number of observations per tree leaf. Default is 1 for classification and 5 for regression.
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1}; 
                    case  lower('NumVariablesToSample')
                            % Number of predictors to select at random for each split
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Hyperparameter Optimization %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    case  lower('OptimizeHyperparameters')
                    % 'none' (default) | 'auto' | 'all'
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1}; 
                    case  lower('HyperparameterOptimizationOptions')
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1}; 
                    case  lower('Standardize')
                        if varargin{i+1}   == true
                            me.dataTBL      = normalize(me.dataTBL,'center','mean','scale','std');
                        end
                    case  lower('Weights')
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};           
                    case  lower('Surrogate')
                            % Surrogate decision splits flag
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};       
                    case  lower('SplitCriterion')
                            % 'MSE' (default)
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};        
                    case  lower('ResponseTransform')
                            % The default is 'none', which means @(y)y, or no transformation.
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};       
                    case  lower('Reproducible')
                            % Flag to enforce reproducibility
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};  
                    case  lower('PruneCriterion')
                            % 'mse' (default)
                            me.varListTree{i  }     = varargin{i  };
                            me.varListTree{i+1}     = varargin{i+1}; 
                     case  lower('Prune')
                            % Flag to estimate optimal sequence of pruned subtrees
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};        
                    case  lower('QuadraticErrorTolerance')
                            % Quadratic error tolerance
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};         
                     case  lower('PredictorSelection')
                            % 'allsplits' : Standard CART — Selects the split predictor that maximizes the split-criterion gain over all possible splits of all predictors [1].
                            % 'curvature' :	Curvature test — Selects the split predictor that minimizes the p-value of chi-square tests of independence between each predictor and the response [2]. Training speed is similar to standard CART.
                            % 'interaction-curvature' : Interaction test — Chooses the split predictor that minimizes the p-value of chi-square tests of independence between each predictor and the response (that is, conducts curvature tests), and that minimizes the p-value of a chi-square test of independence between each pair of predictors and response [2]. Training speed can be slower than standard CART.
                            me.varListTree{i  }     = varargin{i  };
                            me.varListTree{i+1}     = varargin{i+1};        
                    case  lower('NumBins')
                            % Number of bins for numeric predictors
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};        
                    case  lower('MinParentSize')
                            % Minimum number of branch node observations
                            % 10 (default)
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1}; 
                    case  lower('MaxDepth')
                            % Maximum tree depth
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};          
                    case  lower('CategoricalPredictors')
                            % Categorical predictors list
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};      
                    case  lower('MergeLeaves')
                            me.varList{i  }     = varargin{i  };
                            me.varList{i+1}     = varargin{i+1};       
                    case  lower('formula')
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