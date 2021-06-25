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

classdef genfisRL < fitsvmRL
    
    %%%%%%%%%%%%%%%%%%%%%%
    % Neuron Network OOP %
    %%%%%%%%%%%%%%%%%%%%%%

    % List of class attribute
    properties (SetAccess = protected, GetAccess = public)
        clusteringType;     % Method of clustetring
        genOptions;         % MATLAB object of genfisOptions
        tuneOptions;        % MATLAB object of tunefisOptions
    end
    
    
    % Overried the abstract behavior
    methods( Access = public )
                        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to automatically generate the fuzzy rules %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = autoRule(me,numMdl)
            
            % Create routine parameter
            if nargin  <= 2
                % Count number of fields in the given structure
                numMdl  = numel( fieldnames(me.mdl) );
                mdl     = me.mdl.(me.nameList{numMdl});
            end
            
            % Generate the automatic rules
            mdl         = me.autoRuleModel(mdl);
            
            % Return to main
            me.mdl.(me.nameList{numMdl}) = mdl;
            
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to fit the neural network model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        mdlDiagnostic(me,mdl)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to make prediction through the trained model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function pred   = predict(me,dataTest,numMdl)
            if nargin  <= 2
                % Count number of fields in the given structure
                numMdl  = numel( fieldnames(me.mdl) );
            end
            if isempty( dataTest{:,:} ) == false
                pred        = evalfis( me.mdl.(me.nameList{numMdl}), dataTest{:,:} );
            else
                pred        = [];
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = genfisRL(cellFeatureName,strResponseVarName)
            me              = me@fitsvmRL(cellFeatureName,strResponseVarName);
            me.mdlTitle     = "Adaptive Neuro Fuzzy Inference System";
            me.tuneOptions  = tunefisOptions("Method","particleswarm");
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = tuneParameter(me,varargin)
            
            % Create the empty structure of argument input variable 
            me.varList = cell(1,length(varargin(:)));
            
            for i = 1:2:length(varargin)
                
                switch lower(varargin{i})
                    
                    case  lower('Method')
                        % "ga"              : genetic algorithm
                        % "particleswarm"   : particle swarm
                        % "patternsearch"   : pattern search
                        % "simulannealbnd"  : simulated annealing algorithm
                        % "anfis" — adaptive neuro-fuzzy
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                     case  lower('MethodOptions')
                        % options created using optimoptions
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                     case  lower('OptimizationType')
                        % "tuning"          : Optimize the existing input, output, and rule parameters without learning new rules.
                        % "learning"        : Learn new rules up to the maximum number of rules specified by NumMaxRules.
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                     case  lower('NumMaxRules')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                     case  lower('IgnoreInvalidParameters')
                        % options created using optimoptions
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                     case  lower('DistanceMetric')
                        % "rmse"  : Root-mean-squared error
                        % "norm1" : Vector 1-norm
                        % "norm2" : Vector 2-norm
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                     case  lower('DistanceMetric')
                        %  "anfis" tuning method does not support parallel computation
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                     case  lower('KFoldValue')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                     case  lower('ValidationTolerance')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                     case  lower('ValidationWindowSize')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                     case  lower('Display')
                        % "all"             : Display both training and validation results.
                        % "tuningonly"      : Display only training results.
                        % "validationonly"  : Display only validation results.
                        % "none"            : Display neither training nor validation results.
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                end
            
            end
            
            % Filter out the empty cell    
            me.varList = me.varList(~cellfun('isempty',me.varList)); % Call Built-in string
                
            % Check about the existanc of CVPartition object
            if ~isempty(me.c)
                me.chkCV    = true;
                me.flagShow = false;
            end
            
            % Creating the option set for genfis command
            me.tuneOptions  = tunefisOptions(me.varList{:});
            
        end
        
    end
        
    methods( Access = protected )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function mdl = autoRuleModel(me,mdl)
            
            % Creat routine parameters
            numInput    = length( mdl.Inputs  );
            numOutput   = length( mdl.Outputs );
            numElement  = numInput + numOutput + 2;
            numMF       = length( mdl.Inputs(1).MembershipFunctions );
            Rule        = ones( numMF, numElement );
            
            % Auto generate the rule to fuzzy inference system
            for i = 1:numMF
                Rule(i,1:(numInput+numOutput) ) = i;
            end
            
            % Assign the generated rule
            mdl.Rules = [];
            mdl = addRule( mdl, Rule );
                        
        end
        
        function mdl = fitmodel(me,TBL,varname,varargin)
            
            % Extract the input feature and resonse variable 
            idxTBL = TBL.Properties.VariableNames ~= me.ResponseVarName ;
            XData  = TBL{ :,idxTBL};
            YData  = TBL{ :,varname };
            
            % Generate fuzzy inference system object
            mdl    = genfis(XData,YData,me.genOptions);
             
            % Obtain tunable settings from fuzzy inferernce system
            [in,out,rule] = getTunableSettings(mdl);
            
            % Tuning the fuzzy model
            if lower(me.tuneOptions.Method) == lower("anfis")
                paramset = [in;out];
            else
                paramset = [in;out;rule];
            end
            mdl = tunefis(mdl,paramset,XData,YData,me.tuneOptions);
            
            if me.clusteringType ~= lower("GridPartition")
                return
            end
            
            % Setting the name of input membership function
            numInputs    = length(mdl.Inputs);
            numOutputs   = length(mdl.Outputs);
            for i = 1:numInputs
                mdl.Inputs(1,i).Name = me.featureName(i);
                for j = 1:numInputs
                    mdl.Inputs(1,i).Membershipfunctions(1,j).Name = strcat(me.featureName(i),"mf",num2str(j));
                end
                
            end
            
            % Assign the names to the membership function
            for i = 1:numOutputs
                mdl.Outputs(1,i).Name = me.ResponseVarName(i);
                for j = 1:numInputs
                    mdl.Outputs(1,i).Membershipfunctions(1,j).Name = strcat(me.ResponseVarName(i),"mf",num2str(j));
                end
            end
           
                        
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
                    % Clustering Type %
                    %%%%%%%%%%%%%%%%%%%
                    case  lower('ClusteringType')
                        me.clusteringType = varargin{i+1};
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Grid Partitioning Options %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    case  lower('NumMembershipFunctions')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('InputMembershipFunctionType')
                        % 'gbellmf'	Generalized bell-shaped membership function	gbellmf
                        % 'gaussmf'	Gaussian membership function	gaussmf
                        % 'gauss2mf'	Gaussian combination membership function	gauss2mf
                        % 'trimf'	Triangular membership function	trimf
                        % 'trapmf'	Trapezoidal membership function	trapmf
                        % 'sigmf'	Sigmoidal membership function	sigmf
                        % 'dsigmf'	Difference between two sigmoidal membership functions	dsigmf
                        % 'psigmf'	Product of two sigmoidal membership functions	psigmf
                        % 'zmf'	Z-shaped membership function	zmf
                        % 'pimf'	Pi-shaped membership function	pimf
                        % 'smf'	S-shaped membership function	smf
                        % Character vector or string	Name of a custom membership function in the current working folder or on the MATLAB® path	Build Fuzzy Systems Using Custom Functions
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('OutputMembershipFunctionType')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};  
                        
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Subtractive Clustering Options %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                    case  lower('ClusterInfluenceRange')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};
                    case  lower('DataScale')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('SquashFactor')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};     
                    case  lower('AccpetRatio')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};    
                    case  lower('RejectRatio')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('Verbose')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('CustomClusterCenters')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                        
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Fuzzy c-mean (FCM) Clustering Options %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                    case  lower('FISType')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('NumClusters')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};       
                    case  lower('Exponent')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};        
                    case  lower('MaxNumIteration')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};      
                    case  lower('MaxImprovement')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1};   
                    case  lower('Verbose')
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
            
            % Creating the option set for genfis command
            me.genOptions              = genfisOptions(me.clusteringType,me.varList{:});

        end
        
    end

end