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

classdef RegLearner < rootLearner
    
    % List of protected attribute
    properties( Access = protected  )
        % Parameter of the class control process
        pFactor         = [0.15,0.15];  % Array of { pTestData,pValidateData }
        flagShow        = true;         % Show the results in details
        KFolds          = 10;           % Numbe of k-fold cross validation
        chkCV           = false;        % The existance of CVPartision object
        varList         = {};           % The list of routine input arguments
        nameList        = {};           % The name of each k-fold
        c               = [];           % The object of CVPartition
        mdlTitle        = [];           % The methodology name
    end
          
    % List of public attribute 
    properties( Access = public )   
        % List of class attribute
        R2TBL           = [];           % Table of R2
        EvaTBL          = [];           % Table of adjR2
        Residuals       = [];           % Table of model residuals
        ResidualsTest   = [];           % Table of residual testing
    end
    
    % List of protected behavior
    methods( Access = protected )
        resDiagnostic( me , res, ActData, PredData, RMSE, strTitle );
    end
    
    % List of public behavior
    methods( Access = public )
        HypTBL = resHypothesis( me , res, alpha )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % List of class constructor %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function me = RegLearner(cellFeatureName,strResponseVarName)
            me = me@rootLearner(cellFeatureName,strResponseVarName);
        end
        
    end
        
    % List of abstract behavior
    methods( Abstract, Access = public )
        me = fit(me,varargin);
        pred = predict(me,dataTest);
    end
    
    methods( Abstract, Access = protected )
        me = extractVargin(varargin);
    end
    
end