%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% # Released under MIT License %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) July, 2021 Akara Kijkarncharoensin, akara_kij@utcc.ac.th  %
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

classdef fitnetRL < RegLearner
    
    %%%%%%%%%%%%%%%%%%%%%%
    % Neuron Network OOP %
    %%%%%%%%%%%%%%%%%%%%%%
    % List of class attribute
    properties (Access = protected)
        nNeuron         = [10];         % Number of neuron in the hidden layer
        
        % List of training function
        %   'trainlm'   : Levenberg-Marquardt
        %   'trainbr'   : Bayesian Regularization
        %   'trainbfg'  : BFGS Quasi-Newton
        %   'trainrp'   : Resilient Backpropagation
        %   'trainscg'  : Scaled Conjfugate Gradient
        %   'traincgb'  : Conjugate Gradient with Powell/Beale Restarts
        %   'traincgf'  : Fletcher-Powell Conjugate Gradient
        %   'traincgp'  : Polak-Ribiére Conjugate Gradient
        %   'trainoss'  : One Step Secant
        %   'traingdx'  : Variable Learning Rate Gradient Descent
        %   'traingdm'  : Gradient Descent with Momentum
        %   'traingd'   : Gradient Descent
        trainFcn    = 'trainlm'; 
        
        % Network parameters : divideFcn
        % dividerand  : Divide the data randomly (default)
        % divideblock : Divide the data into contiguous blocks
        % divideint   : Divide the data using an interleaved selection
        % divideind   : Divide the data by index
        divideFcn       = 'dividerand'; 
        
        % Assign transfer function of the hidden layer
        % 'tansig' : Hyperbolic tangent sigmoid transfer function
        % 'logsig' : Log-sigmoid transfer function
        % 'radbas' : Radial Basis Neural Networks
        % 'purelin': Linear fucntion
        % Strucuter { 'HiddenLayersFunc' ('OutputLayer.Func') }
        transFcn         = {'tansig'};
        
        % Set Input and Output Pre/Post-Processing Functions
        %   'mapminmax'  : Normalize inputs/targets to fall in the range [−1, 1]
        %   'mapstd'     : Normalize inputs/targets to have zero mean and unity variance
        %   'processpca' : Extract principal components from the input vector
        %   'fixunknowns': Process unknown inputs
        %   'removeconstantrows':removeconstantrows
        inputProcessFcns  = {'removeconstantrows','mapstd'}; 
        outputProcessFcns = {'removeconstantrows','mapstd'};
        
        % Assign the training record
        tr          = [];
    end
          
    % Overried the abstract behavior
    methods( Access = public )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to fit the neural network model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        me = fit(me,varargin);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to make prediction through the trained model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function pred = predict(me,dataTest,numMdl)
            if nargin <= 2
                Count number of fields in the given structure
                numMdl = numel( fieldnames(me.mdl) );
            end
            pred = me.mdl.( me.nameList{numMdl} )(dataTest');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Routine to train the neuron network model %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [mdl,tr] = train(me,mdl,XData,YData)
            [mdl, tr]   = train(mdl,XData,YData,me.varList{:});
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = fitnetRL(cellFeatureName,strResponseVarName)
            me = me@RegLearner(cellFeatureName,strResponseVarName);
            me.mdlTitle = 'Neuron Network';
        end
    end
    
    methods( Access = protected )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Engine of the machine learner %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function mdl = fitmodel(me,XData,YData)
            % Create the model of the neural network
            mdl = fitnet(me.nNeuron, me.trainFcn);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Setting the parameter of the neural network %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The routine to extract the name-value parameter %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = extractVargin(me,varargin)
            
            for i = 1:2:length(varargin)

                switch lower(varargin{i})
                    case  lower('showResources')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('useGPU')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('useParallel')
                        me.varList{i  }     = varargin{i  };
                        me.varList{i+1}     = varargin{i+1}; 
                    case  lower('nNeuron')
                        me.nNeuron          = varargin{i+1};
                    case  lower('inputProcessFcns')
                        me.inputProcessFcns = varargin{i+1};
                    case  lower('outputProcessFcns')
                        me.outputProcessFcns= varargin{i+1};
                    case  lower('trainFcn')
                        me.trainFcn         = varargin{i+1};
                    case  lower('kFold')
                        me.KFolds           = varargin{i+1};
                        me.c = cvpartition(n.All,'KFold',Kfolds);
                    case  lower('Holdout')
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Network parameters : divideFcn %
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % dividerand   : Divide the data randomly (default)
                        % divideblock  : Divide the data into continuous blocks
                        % divideint    : Divide the data using an interleaved selection
                        me.KFolds           = 1;
                        me.chkCV            = false;
                        me.pFactor          = varargin{i+1};
                    case  lower('Leaveout')
                        if varargin{i+1} == true
                            me.c = cvpartition(n.All,'LeaveOut');
                        end
                    case  lower('Resubstitution')
                        if varargin{i+1} == true
                            % dividetrain : Assign all targets to training set
                            me.KFolds       = 1;
                            me.divideFcn    = 'dividetrain';   
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
            for fold=1:me.KFolds
               me.nameList{fold} = "fold"+num2str(fold);
            end
            me.nameList{end}     = "CrossVal";  % Name of k-fold
            me.R2TBL             = [];          % R2 table
            me.EvaTBL            = [];          % Evaluation table
            me.Residuals         = [];          % Residual table
        end
    end
    
end
