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

function me = fit(me,varargin)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit model by the linear model %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % The routine structure variable
    n           = struct('Train' ,0 , 'Val',0    , 'Test',0 , 'CrossVal',10   , 'Factor'  ,0, 'All',0);
    sumCrossVal = struct('Step'  ,struct('AE' ,[], 'APE' ,[], 'BE'      ,[]   , 'SE'      ,[] ) , ...
                         'Robust',struct('AE' ,[], 'APE' ,[], 'BE'      ,[]   , 'SE'      ,[] ) );
    idx         = struct('All'   ,[], 'Train' ,[], 'Val' ,[], 'Test'    ,[]   , 'CrossVal',[]   );
    TBL         = idx;  % The table of data
    Eva         = struct('Step'  ,idx,'Robust', idx);  % Evaluation results
    R2          = Eva;  % The R Square
    adjR2       = Eva;  % The adjusted R Square
    avg         = Eva;  % The average value
    Diff        = Eva;  % The sum of total error
    Res         = Eva;  % The model resiudals 
    Pred        = Eva;  % The model predictions
    MAE         = Eva;  % The mean absolute value
    APE         = Eva;  % The mean absolute proportional error
    MBE         = Eva;  % The mean bias error
    MSE         = Eva;  % The mean square error
    RMSE        = Eva;  % The root of mean square error
    MSEANOVA    = Eva;  % The mean square error of ANOVA
    RMSEANOVA   = Eva;  % The root mean square error of ANOVA
    HYP         = Eva;  % The normality test of the residuals
    
    % The anchor state of the random generation 
    me.sRand    = RandStream('mcg16807','Seed',144);

    % The number of each data set
    ResponseVar = me.ResponseVarName;
    idxTBL      = me.dataTBL.Properties.VariableNames ~= ResponseVar ;
    TBL.All     = me.dataTBL;
    
    % Filter the data to the focus one
    if isnumeric(me.dataID) && isvector(me.dataID)
        TBL.All  = TBL.All(me.dataID,:);
    end
    
    % Devide the data into separate group
    n.All       = size(TBL.All,1);
    n.Factor    = size(TBL.All,2)-1; % [ FC VM A HHV ]
    NData       = [1:n.All];
    n.Train     = n.All;
    n.Val       = 0;
    n.Test      = 0;
    
    % Extract the model Name-Value parameters
    me.extractVargin(varargin{:});
 
    % Memory Allocation
    n.CrossVal              = 0;
    
    sumCrossVal.Step.AE     = 0.;
    sumCrossVal.Step.APE    = 0.;
    sumCrossVal.Step.BE     = 0.;
    sumCrossVal.Step.SE     = 0.;
    CrossValData.Step       = TBL.All.(ResponseVar);
    
    sumCrossVal.Robust.AE   = 0.;
    sumCrossVal.Robust.APE  = 0.;
    sumCrossVal.Robust.BE   = 0.;
    sumCrossVal.Robust.SE   = 0.;
    CrossValData.Robust     = TBL.All.(ResponseVar);
    
    % Perform K-fold evaluation
    for fold=1:me.KFolds
        
        if me.chkCV
           %
           % Leaveout ,KFold & CVPartition
           % 
           switch  lower(me.c.Type)
               case  lower('Holdout')
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Network parameters : divideFcn %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % dividerand  : Divide the data randomly (default)
                    % divideblock : Divide the data into contiguous blocks
                    % divideint   : Divide the data using an interleaved selection
                    idx.Train = NData( training(me.c) );
                    idx.Val   = [];
                    idx.Test  = NData( test(me.c)     );
               otherwise
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Network parameters : divideFcn %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % divideind   : Divide the data by index
                    % dividetrain : Assign all targets to training set
                    idx.Train = NData( training(me.c,fold) );
                    idx.Val   = [];
                    idx.Test  = NData( test    (me.c,fold) );
           end
           % The number of each data set
                n.Train       = length(idx.Train);
                n.Val         = length(idx.Val);
                n.Test        = length(idx.Test);
        else
           if ~isempty( me.pFactor )
                % Case of holdout
                n.Test        = round     ( me.pFactor(1)*n.All );
                n.Val         = round     ( me.pFactor(2)*n.All );
                n.Train       = n.All -   ( n.Test       +n.Val );
                idx.Test      = datasample( me.sRand,NData   ,n.Test,'Replace',false );
                NDataSub      = setdiff   ( NData,idx.Test   );
                idx.Val       = datasample( me.sRand,NDataSub,n.Val ,'Replace',false);
                idx.Train     = setdiff   ( NDataSub,idx.Val );
           else
                % Case of resubstitution
                idx.Test      = [];
                idx.Val       = [];
                idx.Train     = NData(:);
           end
        end
        
         % Separate data into three categories: Train, Validate and Testing
        TBL.Test         = TBL.All(idx.Test ,:);
        TBL.Val          = TBL.All(idx.Val  ,:);
        TBL.Train        = TBL.All(idx.Train,:);
        
        % MSE of the linear model compute by divide N-(Factor+1)
        fprintf("Process of the stepwise regression : \n");
        fprintf("================================== \n");
        mdlStep          = stepwiselm(TBL.Train,'constant','ResponseVar',ResponseVar,me.varList{:});
        mdlRobust        = fitlm(TBL.Train,mdlStep.Formula,'ResponseVar',ResponseVar,'RobustOpts','on');
                
        % Collect the k-fold model
        me.mdl.Step.(me.nameList{fold})  = mdlStep;
        me.mdl.Robust.(me.nameList{fold})= mdlRobust;
        
        % Make the predicions ( Use matlab's function )
        Pred.Step.Train  = predict( mdlStep  , TBL.Train(:,idxTBL) );
        Pred.Step.Val    = predict( mdlStep  , TBL.Val  (:,idxTBL) );
        Pred.Step.Test   = predict( mdlStep  , TBL.Test (:,idxTBL) );
        Pred.Step.All    = predict( mdlStep  , TBL.All  (:,idxTBL) );
        Pred.Robust.Train= predict( mdlRobust, TBL.Train(:,idxTBL) );
        Pred.Robust.Val  = predict( mdlRobust, TBL.Val  (:,idxTBL) );
        Pred.Robust.Test = predict( mdlRobust, TBL.Test (:,idxTBL) );
        Pred.Robust.All  = predict( mdlRobust, TBL.All  (:,idxTBL) );
                
        % The prediction error
        Res.Step.Train   = TBL.Train.(ResponseVar) - Pred.Step.Train  ;
        Res.Step.Val     = TBL.Val.  (ResponseVar) - Pred.Step.Val    ;
        Res.Step.Test    = TBL.Test. (ResponseVar) - Pred.Step.Test   ;
        Res.Step.All     = TBL.All.  (ResponseVar) - Pred.Step.All    ;
        Res.Robust.Train = TBL.Train.(ResponseVar) - Pred.Robust.Train;
        Res.Robust.Val   = TBL.Val.  (ResponseVar) - Pred.Robust.Val  ;
        Res.Robust.Test  = TBL.Test. (ResponseVar) - Pred.Robust.Test ;
        Res.Robust.All   = TBL.All.  (ResponseVar) - Pred.Robust.All  ;
        
        % The average value of every data set
        avg.Train        = mean(TBL.Train.(ResponseVar));
        avg.Val          = mean(TBL.Val.  (ResponseVar));
        avg.Test         = mean(TBL.Test. (ResponseVar));
        avg.All          = mean(TBL.All.  (ResponseVar));
        
        Diff.Train       = TBL.Train.(ResponseVar) - avg.Train;
        Diff.Val         = TBL.Val. (ResponseVar)  - avg.Val; 
        Diff.Test        = TBL.Test.(ResponseVar)  - avg.Test; 
        Diff.All         = TBL.All. (ResponseVar)  - avg.All;
        
        if me.flagShow

            % Diagonostic plot
            figure; plotDiagnostics(mdlRobust);
            figure; plotDiagnostics(mdlRobust ,'cookd'      );

            % Residual plot
            figure; plotResiduals  (mdlRobust ,'histogram'  );
            figure; plotResiduals  (mdlRobust ,'lagged'     );
            figure; plotResiduals  (mdlRobust ,'fitted'     );
            figure; plotResiduals  (mdlRobust ,'probability');

        end
                
        % Compute the paramber of the whole KFolds
        n.CrossVal              = n.CrossVal             + n.Test;
        sumCrossVal.Step.AE     = sumCrossVal.Step.AE    + sum ( abs ( Res.Step.Test  )  );
        sumCrossVal.Step.APE    = sumCrossVal.Step.APE   + sum ( abs ( Res.Step.Test  )./TBL.Test.(ResponseVar) );
        sumCrossVal.Step.BE     = sumCrossVal.Step.BE    + sum (     ( Res.Step.Test  )./TBL.Test.(ResponseVar) );
        sumCrossVal.Step.SE     = sumCrossVal.Step.SE    + sum ( Res.Step.Test.*Res.Step.Test );
        CrossValData.Step(idx.Test) = Pred.Step.Test;
        
        sumCrossVal.Robust.AE   = sumCrossVal.Robust.AE  + sum ( abs ( Res.Robust.Test ) );
        sumCrossVal.Robust.APE  = sumCrossVal.Robust.APE + sum ( abs ( Res.Robust.Test )./TBL.Test.(ResponseVar));
        sumCrossVal.Robust.BE   = sumCrossVal.Robust.BE  + sum (     ( Res.Robust.Test )./TBL.Test.(ResponseVar));
        sumCrossVal.Robust.SE   = sumCrossVal.Robust.SE  + sum ( Res.Robust.Test.*Res.Robust.Test );
        CrossValData.Robust(idx.Test) = Pred.Robust.Test;
        
        % Compute the SE of the interval estimation 
        % and compute the mean of the error square
        MAE.Step.All       = sum ( abs( Res.Step.All )                 )/ n.All;  
        APE.Step.All       = sum ( abs( Res.Step.All )./TBL.All.(ResponseVar) )/ n.All;  
        MBE.Step.All       = sum (    ( Res.Step.All )./TBL.All.(ResponseVar) )/ n.All; 
        MSE.Step.All       = sum (      Res.Step.All.*Res.Step.All     )/ n.All;  
        RMSE.Step.All      = sqrt( MSE.Step.All      );  
        HYP.Step.All       = me.resHypothesis( Res.Step.All );
        
        MAE.Step.Train     = sum ( abs( Res.Step.Train )                   )/ n.Train;  
        APE.Step.Train     = sum ( abs( Res.Step.Train )./TBL.Train.(ResponseVar))/ n.Train;  
        MBE.Step.Train     = sum (    ( Res.Step.Train )./TBL.Train.(ResponseVar))/ n.Train; 
        MSE.Step.Train     = sum (      Res.Step.Train.*Res.Step.Train     )/ n.Train;  
        RMSE.Step.Train    = sqrt( MSE.Step.Train      ); 
        HYP.Step.Train     = me.resHypothesis( Res.Step.Train );
        
        if( n.Val ~= 0 )
            MAE.Step.Val     = sum ( abs( Res.Step.Val )                     )/ n.Val;  
            APE.Step.Val     = sum ( abs( Res.Step.Val ) ./TBL.Val.(ResponseVar)   )/ n.Val;  
            MBE.Step.Val     = sum (    ( Res.Step.Val ) ./TBL.Val.(ResponseVar)   )/ n.Val; 
            MSE.Step.Val     = sum (      Res.Step.Val.*Res.Step.Val         )/ n.Val; 
            RMSE.Step.Val    = sqrt( MSE.Step.Val      ); 
            HYP.Step.Val     = me.resHypothesis( Res.Step.Train );
        else
            MAE.Step.Val     = missing;  
            APE.Step.Val     = missing;
            MBE.Step.Val     = missing;
            MSE.Step.Val     = missing;
            RMSE.Step.Val    = missing;
            HYP.Step.Val     = missing;
        end

        MAE.Step.Test      = sum ( abs( Res.Step.Test )                    )/ n.Test;  
        APE.Step.Test      = sum ( abs( Res.Step.Test )./TBL.Test.(ResponseVar)  )/ n.Test;  
        MBE.Step.Test      = sum (    ( Res.Step.Test )./TBL.Test.(ResponseVar)  )/ n.Test; 
        MSE.Step.Test      = sum ( Res.Step.Test.*Res.Step.Test            )/ n.Test;      
        RMSE.Step.Test     = sqrt( MSE.Step.Test      ); 
        HYP.Step.Test      = me.resHypothesis( Res.Step.Test  );
                
        MAE.Robust.All     = sum ( abs( Res.Robust.All )                   )/ n.All;  
        APE.Robust.All     = sum ( abs( Res.Robust.All )./TBL.All.(ResponseVar)   )/ n.All;  
        MBE.Robust.All     = sum (    ( Res.Robust.All )./TBL.All.(ResponseVar)   )/ n.All; 
        MSE.Robust.All     = sum (      Res.Robust.All.*Res.Robust.All     )/ n.All;  
        RMSE.Robust.All    = sqrt( MSE.Robust.All      );
        HYP.Robust.All     = me.resHypothesis( Res.Robust.All );
        
        MAE.Robust.Train   = sum ( abs( Res.Robust.Train )                   )/ n.Train;  
        APE.Robust.Train   = sum ( abs( Res.Robust.Train )./TBL.Train.(ResponseVar))/ n.Train;  
        MBE.Robust.Train   = sum (    ( Res.Robust.Train )./TBL.Train.(ResponseVar))/ n.Train; 
        MSE.Robust.Train   = sum (      Res.Robust.Train.*Res.Robust.Train   )/ n.Train;  
        RMSE.Robust.Train  = sqrt( MSE.Robust.Train    );  
        HYP.Robust.Train   = me.resHypothesis( Res.Robust.Train );
        
        if( n.Val ~= 0 )
            MAE.Robust.Val    = sum ( abs( Res.Robust.Val )                    )/ n.Val;  
            APE.Robust.Val    = sum ( abs( Res.Robust.Val ) ./TBL.Val.(ResponseVar)  )/ n.Val;  
            MBE.Robust.Val    = sum (    ( Res.Robust.Val ) ./TBL.Val.(ResponseVar)  )/ n.Val; 
            MSE.Robust.Val    = sum (      Res.Robust.Val.*Res.Robust.Val      )/ n.Val; 
            RMSE.Robust.Val   = sqrt( MSE.Robust.Val      );
            HYP.Robust.Val    = me.resHypothesis( Res.Robust.Val );
        else
            MAE.Robust.Val    = missing;  
            APE.Robust.Val    = missing;
            MBE.Robust.Val    = missing;
            MSE.Robust.Val    = missing;
            RMSE.Robust.Val   = missing;
            HYP.Robust.Val    = missing;
        end
                       
        MAE.Robust.Test       = sum ( abs( Res.Robust.Test )                   )/ n.Test;  
        APE.Robust.Test       = sum ( abs( Res.Robust.Test )./TBL.Test.(ResponseVar) )/ n.Test;  
        MBE.Robust.Test       = sum (    ( Res.Robust.Test )./TBL.Test.(ResponseVar) )/ n.Test; 
        MSE.Robust.Test       = sum ( Res.Robust.Test.*Res.Robust.Test         )/n.Test;      
        RMSE.Robust.Test      = sqrt( MSE.Robust.Test  );        
        HYP.Robust.Test       = me.resHypothesis( Res.Robust.Test  );
                
        MSEANOVA.Step.All     = sum ( Res.Step.All  .*Res.Step.All   )/( n.All   - n.Factor - 1);      
        MSEANOVA.Step.Train   = sum ( Res.Step.Train.*Res.Step.Train )/( n.Train - n.Factor - 1);      
        MSEANOVA.Step.Val     = sum ( Res.Step.Val  .*Res.Step.Val   )/( n.Val   - n.Factor - 1);      
        MSEANOVA.Step.Test    = sum ( Res.Step.Test .*Res.Step.Test  )/( n.Test  - n.Factor - 1);      
        RMSEANOVA.Step.All    = sqrt( MSEANOVA.Step.All   ); 
        RMSEANOVA.Step.Train  = sqrt( MSEANOVA.Step.Train );
        RMSEANOVA.Step.Val    = sqrt( MSEANOVA.Step.Val   );  
        RMSEANOVA.Step.Test   = sqrt( MSEANOVA.Step.Test  ); 
        
        MSEANOVA.Robust.All   = missing;      
        MSEANOVA.Robust.Train = mdlRobust.MSE;      
        MSEANOVA.Robust.Val   = missing;      
        MSEANOVA.Robust.Test  = missing;      
        RMSEANOVA.Robust.All  = missing; 
        RMSEANOVA.Robust.Train= mdlRobust.RMSE;
        RMSEANOVA.Robust.Val  = missing;  
        RMSEANOVA.Robust.Test = missing; 
        
        % Compute the regression coefficient
        R2.Step.Train     = 1 - sum( Res.Step.Train.*Res.Step.Train )  ...
                              / sum( Diff.Train    .*Diff.Train     );
        R2.Step.Val       = 1 - sum( Res.Step.Val  .*Res.Step.Val   ) ...
                              / sum( Diff.Val      .*Diff.Val       );
        R2.Step.Test      = 1 - sum( Res.Step.Test .*Res.Step.Test  ) ...
                              / sum( Diff.Test     .*Diff.Test      );
        R2.Step.All       = 1 - sum( Res.Step.All  .*Res.Step.All   ) ...
                              / sum( Diff.All      .*Diff.All       );
                    
        R2.Robust.Train   = 1 - sum( Res.Robust.Train.*Res.Robust.Train )  ...
                              / sum( Diff.Train      .*Diff.Train       );
        R2.Robust.Val     = 1 - sum( Res.Robust.Val  .*Res.Robust.Val   ) ...
                              / sum( Diff.Val        .*Diff.Val         );
        R2.Robust.Test    = 1 - sum( Res.Robust.Test .*Res.Robust.Test  ) ...
                              / sum( Diff.Test       .*Diff.Test        );
        R2.Robust.All     = 1 - sum( Res.Robust.All  .*Res.Robust.All   ) ...
                              / sum( Diff.All        .*Diff.All         );

        % Compute the adjusted R^2
        adjR2.Step.Train  = 1 - (1-R2.Step.Train)*( n.Train - 1)/( n.Train - n.Factor - 1);
        adjR2.Step.Val    = 1 - (1-R2.Step.Val  )*( n.Val   - 1)/( n.Val   - n.Factor - 1);
        adjR2.Step.Test   = 1 - (1-R2.Step.Test )*( n.Test  - 1)/( n.Test  - n.Factor - 1);
        adjR2.Step.All    = 1 - (1-R2.Step.All  )*( n.All   - 1)/( n.All   - n.Factor - 1); 
        
        adjR2.Robust.Train= 1 - (1-R2.Robust.Train)*( n.Train - 1)/( n.Train - n.Factor - 1);
        adjR2.Robust.Val  = 1 - (1-R2.Robust.Val  )*( n.Val   - 1)/( n.Val   - n.Factor - 1);
        adjR2.Robust.Test = 1 - (1-R2.Robust.Test )*( n.Test  - 1)/( n.Test  - n.Factor - 1);
        adjR2.Robust.All  = 1 - (1-R2.Robust.All  )*( n.All   - 1)/( n.All   - n.Factor - 1);
                
        % Summary the model performance
        varNames   = {'R^2','adj.R^2'};
        rowNames   = {'Train','Validate','Test','All'};
        R2Data     = [R2.Step.Train   ;R2.Step.Val   ;R2.Step.Test   ;R2.Step.All   ];
        adjR2Data  = [adjR2.Step.Train;adjR2.Step.Val;adjR2.Step.Test;adjR2.Step.All];
        me.R2TBL.Step.( me.nameList{fold} )   = table(R2Data, adjR2Data , 'VariableNames', varNames, 'RowNames', rowNames );
        
        varNames   = {'R^2','adj.R^2'};
        rowNames   = {'Train','Validate','Test','All'};
        R2Data     = [R2.Robust.Train   ;R2.Robust.Val   ;R2.Robust.Test   ;R2.Robust.All   ];
        adjR2Data  = [adjR2.Robust.Train;adjR2.Robust.Val;adjR2.Robust.Test;adjR2.Robust.All];
        me.R2TBL.Robust.( me.nameList{fold} ) = table(R2Data,adjR2Data , 'VariableNames', varNames, 'RowNames', rowNames );
        
        varNames   = {'Train','Validate','Test','All'};
        rowNames   = {'MAE','APE','MBE','MSE','RMSE','MSEANOVA','RMSEANOVA'};
        Eva.Train  = [ MAE.Step.Train;APE.Step.Train;MBE.Step.Train;MSE.Step.Train;RMSE.Step.Train;MSEANOVA.Step.Train;RMSEANOVA.Step.Train ];
        Eva.Val    = [ MAE.Step.Val  ;APE.Step.Val  ;MBE.Step.Val  ;MSE.Step.Val  ;RMSE.Step.Val  ;MSEANOVA.Step.Val  ;RMSEANOVA.Step.Val   ];
        Eva.Test   = [ MAE.Step.Test ;APE.Step.Test ;MBE.Step.Test ;MSE.Step.Test ;RMSE.Step.Test ;MSEANOVA.Step.Test ;RMSEANOVA.Step.Test  ];
        Eva.All    = [ MAE.Step.All  ;APE.Step.All  ;MBE.Step.All  ;MSE.Step.All  ;RMSE.Step.All  ;MSEANOVA.Step.All  ;RMSEANOVA.Step.All   ];
        me.EvaTBL.Step.( me.nameList{fold} )  = table(Eva.Train ,Eva.Val ,Eva.Test ,Eva.All ,'VariableNames', varNames, 'RowNames', rowNames );
       
        varNames   = {'Train','Validate','Test','All'};
        rowNames   = {'MAE','APE','MBE','MSE','RMSE','MSEANOVA','RMSEANOVA'};
        Eva.Train  = [ MAE.Robust.Train;APE.Robust.Train;MBE.Robust.Train;MSE.Robust.Train;RMSE.Robust.Train;MSEANOVA.Robust.Train;RMSEANOVA.Robust.Train ];
        Eva.Val    = [ MAE.Robust.Val  ;APE.Robust.Val  ;MBE.Robust.Val  ;MSE.Robust.Val  ;RMSE.Robust.Val  ;MSEANOVA.Robust.Val  ;RMSEANOVA.Robust.Val   ];
        Eva.Test   = [ MAE.Robust.Test ;APE.Robust.Test ;MBE.Robust.Test ;MSE.Robust.Test ;RMSE.Robust.Test ;MSEANOVA.Robust.Test ;RMSEANOVA.Robust.Test  ];
        Eva.All    = [ MAE.Robust.All  ;APE.Robust.All  ;MBE.Robust.All  ;MSE.Robust.All  ;RMSE.Robust.All  ;MSEANOVA.Robust.All  ;RMSEANOVA.Robust.All   ];
        me.EvaTBL.Robust.( me.nameList{fold}) = table(Eva.Train ,Eva.Val ,Eva.Test ,Eva.All ,'VariableNames', varNames, 'RowNames', rowNames );
        
        varNames   = {'Train','Validate','Test','All'};
        me.Residuals.Step.( me.nameList{fold} ).Data        = { Res.Step.Train, Res.Step.Val, Res.Step.Test, Res.Step.All };
        me.Residuals.Step.( me.nameList{fold} ).Name        = varNames;
        
        varNames   = {'Train','Validate','Test','All'};
        me.Residuals.Robust.( me.nameList{fold} ).Data      = { Res.Robust.Train, Res.Robust.Val, Res.Robust.Test, Res.Robust.All };
        me.Residuals.Robust.( me.nameList{fold} ).Name      = varNames;
        
        me.ResidualsTest.Step.( me.nameList{fold} ).Train   = HYP.Step.Train;
        me.ResidualsTest.Step.( me.nameList{fold} ).Val     = HYP.Step.Val;
        me.ResidualsTest.Step.( me.nameList{fold} ).Test    = HYP.Step.Test;
        me.ResidualsTest.Step.( me.nameList{fold} ).All     = HYP.Step.All;
        
        me.ResidualsTest.Robust.( me.nameList{fold} ).Train = HYP.Robust.Train;
        me.ResidualsTest.Robust.( me.nameList{fold} ).Val   = HYP.Robust.Val;
        me.ResidualsTest.Robust.( me.nameList{fold} ).Test  = HYP.Robust.Test;
        me.ResidualsTest.Robust.( me.nameList{fold} ).All   = HYP.Robust.All;
                    
        fprintf("\n");
        fprintf("Summary of the linear model : \n");
        fprintf("================================== \n");
        fprintf("R^2 : %s \n", me.nameList{fold} );
        fprintf("Stepwise Regression : \n")
        disp(me.R2TBL.Step.(me.nameList{fold}) );
        fprintf("Robust Regression : \n")
        disp(me.R2TBL.Robust.(me.nameList{fold}) );
        fprintf("ANOVA : \n")
        fprintf("Stepwise regression : \n")
        disp(mdlStep.Coefficients);
        disp(anova(mdlStep,'summary'));

        fprintf("Robust regression : \n")
        disp(mdlRobust.Coefficients);
        disp(anova(mdlRobust,'summary'));
             
    end

    if me.chkCV == false || strcmp(lower(me.c.Type),lower('Holdout'))
        return;
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The summary of K-Fold Statistic %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The average error based on the cross validation
    MAE.Step.CrossVal        = sumCrossVal.Step.AE  /  n.CrossVal;
    APE.Step.CrossVal        = sumCrossVal.Step.APE /  n.CrossVal;
    MBE.Step.CrossVal        = sumCrossVal.Step.BE  /  n.CrossVal;
    MSE.Step.CrossVal        = sumCrossVal.Step.SE  /  n.CrossVal;
    RMSE.Step.CrossVal       = sqrt( MSE.Step.CrossVal );
    MSEANOVA.Step.CrossVal   = sumCrossVal.Step.SE  /( n.CrossVal-n.Factor-1 );
    RMSEANOVA.Step.CrossVal  = sqrt( MSEANOVA.Step.CrossVal  );
    
    MAE.Robust.CrossVal      = sumCrossVal.Robust.AE /  n.CrossVal;
    APE.Robust.CrossVal      = sumCrossVal.Robust.APE/  n.CrossVal;
    MBE.Robust.CrossVal      = sumCrossVal.Robust.BE /  n.CrossVal;
    MSE.Robust.CrossVal      = sumCrossVal.Robust.SE /  n.CrossVal;
    RMSE.Robust.CrossVal     = sqrt( MSE.Robust.CrossVal );
    MSEANOVA.Robust.CrossVal = sumCrossVal.Robust.SE /( n.CrossVal-n.Factor-1 );
    RMSEANOVA.Robust.CrossVal= sqrt( MSEANOVA.Step.CrossVal  );
    
    varNames   =  {'CrossVal'};
    rowNames   =  {'MAE','APE','MBE','MSE','RMSE','MSEANOVA','RMSEANOVA'};
    Eva.CrossVal    = [ MAE.Step.CrossVal;APE.Step.CrossVal;MBE.Step.CrossVal;MSE.Step.CrossVal;RMSE.Step.CrossVal;MSEANOVA.Robust.CrossVal;RMSEANOVA.Robust.CrossVal ];
    me.EvaTBL.Step.( me.nameList{end} )   = table(Eva.CrossVal ,'VariableNames',varNames,'RowNames',rowNames );

    varNames   = {'CrossVal'};
    rowNames   = {'MAE','APE','MBE','MSE','RMSE','MSEANOVA','RMSEANOVA'};
    Eva.CrossVal    = [ MAE.Robust.CrossVal;APE.Robust.CrossVal;MBE.Robust.CrossVal;MSE.Robust.CrossVal;RMSE.Robust.CrossVal;MSEANOVA.Robust.CrossVal;RMSEANOVA.Robust.CrossVal ];
    me.EvaTBL.Robust.( me.nameList{end} ) = table(Eva.CrossVal ,'VariableNames',varNames,'RowNames',rowNames );

    
    % The R^2 of the cross validation
    Res.Step.CrossVal     = TBL.All.(ResponseVar) - CrossValData.Step;
    R2.Step.CrossVal      = 1 - sum( Res.Step.CrossVal.*Res.Step.CrossVal ) ...
                              / sum( Diff.All.*Diff.All );
    adjR2.Step.CrossVal   = 1 - (1-R2.Step.CrossVal  )*( n.CrossVal - 1)/( n.CrossVal - n.Factor - 1);
    
    Res.Robust.CrossVal   = TBL.All.(ResponseVar) - CrossValData.Robust;
    R2.Robust.CrossVal    = 1 - sum( Res.Robust.CrossVal.*Res.Robust.CrossVal ) ...
                              / sum( Diff.All.*Diff.All );
    adjR2.Robust.CrossVal = 1 - (1-R2.Robust.CrossVal)*( n.CrossVal - 1)/( n.CrossVal - n.Factor - 1);
        
    varNames              = {'R^2','adj.R^2'};
    rowNames              = {'CrossVal'};
    R2Data                = [    R2.Step.CrossVal ];
    adjR2Data             = [ adjR2.Step.CrossVal ];
    me.R2TBL.Step.( me.nameList{end}  ) = table( R2Data,adjR2Data,'VariableNames',varNames,'RowNames',rowNames );
    
    varNames              = {'R^2','adj.R^2'};
    rowNames              = {'CrossVal'};
    R2Data                = [    R2.Robust.CrossVal ];
    adjR2Data             = [ adjR2.Robust.CrossVal ];
    me.R2TBL.Robust.( me.nameList{end}) = table( R2Data,adjR2Data,'VariableNames',varNames,'RowNames',rowNames );
    
    varNames              = {'CrossVal'};
    me.Residuals.Step.( me.nameList{end} ).Data   = Res.Step.CrossVal;
    me.Residuals.Step.( me.nameList{end} ).Name   = varNames;

    varNames              = {'CrossVal'};
    me.Residuals.Robust.( me.nameList{end} ).Data = Res.Robust.CrossVal;
    me.Residuals.Robust.( me.nameList{end} ).Name = varNames;
    
    fprintf("Summary of the linear model : \n");
    fprintf("================================== \n");
    fprintf("The validation approach : %s %i \n",me.c.Type, me.c.NumTestSets);
    fprintf("The stepwise regression : \n");
    disp(me.R2TBL.Step.   ( me.nameList{end} ));
    disp(me.EvaTBL.Step.  ( me.nameList{end} ));
    fprintf("The robust regression : \n");
    disp(me.R2TBL.Robust. ( me.nameList{end} ));
    disp(me.EvaTBL.Robust.( me.nameList{end} ));

    % The residual diagnostic
    me.resDiagnostic( Res.Step.CrossVal  , TBL.All.(ResponseVar), CrossValData.Step  , RMSEANOVA.Step.CrossVal  , "The residual of cross validation stepwise model" );
    me.resDiagnostic( Res.Robust.CrossVal, TBL.All.(ResponseVar), CrossValData.Robust, RMSEANOVA.Robust.CrossVal, "The residual of cross validation robust model"   );
    
    % Perform the normaility testing of the residual
    HYP.Step.CrossVal                = me.resHypothesis( Res.Step.CrossVal   );
    HYP.Robust.CrossVal              = me.resHypothesis( Res.Robust.CrossVal );
    
    % Return the results of normality test
    me.ResidualsTest.Step.CrossVal   = HYP.Step.CrossVal;
    me.ResidualsTest.Robust.CrossVal = HYP.Step.CrossVal;
    
end