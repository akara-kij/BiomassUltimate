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

classdef fitensembleRL < fitsvmRL
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit the resemble trees %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % List of class attribute
    properties (Access = protected)
        % List of class attribute
        varListTree;
    end
    
    methods

    end
    
    % Overried the abstract behavior
    methods( Access = public )
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = fitensembleRL(cellFeatureName,strResponseVarName)
            me = me@fitsvmRL(cellFeatureName,strResponseVarName);
            me.mdlTitle = "Ensemble Trees";
        end
    end
    
    methods( Access = protected )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function mdl = fitmodel(me,TBL,varname,varargin)
            if ~isempty(me.varListTree)
                tr  = templateTree(me.varListTree{:});
                mdl = fitrensemble(TBL,varname,'Learners',tr,varargin{:});
            else
                mdl = fitrensemble(TBL,varname,varargin{:});
            end
            me.mdlTitle = "Ensemble trees";
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList      = cell(1,length(varargin(:)));
            me.varListTree  = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    case  lower('Standardize')
                        if varargin{i+1}   == true
                            me.dataTBL      = normalize(me.dataTBL,'center','mean','scale','std');
                        end
                    case  lower('Weights')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};           
                    case  lower('Surrogate')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};       
                    case  lower('SplitCriterion')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};        
                    case  lower('ResponseTransform')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};       
                    case  lower('Reproducible')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};  
                     
                    case  lower('Prune')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};         
                   case  lower('MinParentSize')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1}; 
                    case  lower('MaxDepth')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};          
                    case  lower('CategoricalPredictors')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};      
                    case  lower('MergeLeaves')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};       
                    case  lower('formula')
                        me.varListTree{i  }     = varargin{i  };
                        me.varListTree{i+1}     = varargin{i+1};
                    case  lower('Method')
                        % 'LSBoost' : Least-squares boosting (LSBoost)
                        % 'Bag'     : Bootstrap aggregation (bagging, for example, random forest[2])
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('NumLearningCycles')
                        % Number of tree in random forest , Method = 'Bag'
                        % 100 (default) , Parameter : NLearn
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};          
                    case  lower('NumBins')
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
            me.varList     = me.varList    (~cellfun('isempty',me.varList    )); % Call Built-in string
            me.varListTree = me.varListTree(~cellfun('isempty',me.varListTree)); % Call Built-in string
            
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