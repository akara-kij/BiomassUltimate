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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit model by the ridge regression model %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % The routine structure variable
    n           = struct('Train' ,0 , 'Val'   ,0 , 'Test',0 , 'CrossVal',10   , 'Factor'  ,0, 'All',0);
    
    sumCrossVal = struct('AE'    ,[], 'APE'   ,[], 'BE'  ,[], 'SE'      ,[] );
    idx         = struct('All'   ,[], 'Train' ,[], 'Val' ,[], 'Test'    ,[]   , 'CrossVal',[]   );
    TBL         = idx;  % The table of data
    Eva         = idx;  % Evaluation results
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
    n.CrossVal         = 0;
    
    sumCrossVal.AE     = 0.;
    sumCrossVal.APE    = 0.;
    sumCrossVal.BE     = 0.;
    sumCrossVal.SE     = 0.;
    CrossValData       = TBL.All. (ResponseVar);

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
                n.Test        = round   ( me.pFactor(1)*n.All );
                n.Val         = round   ( me.pFactor(2)*n.All );
                n.Train       = n.All - ( n.Test       +n.Val );
                idx.Test      = datasample( me.sRand,NData   ,n.Test,'Replace',false );
                NDataSub      = setdiff   ( NData,idx.Test );
                idx.Val       = datasample( me.sRand,NDataSub,n.Val ,'Replace',false);
                idx.Train     = setdiff( NDataSub,idx.Val  );
           else
                % Case of resubstitution
                idx.Test      = [];
                idx.Val       = [];
                idx.Train     = NData(:);
           end
        end
        
         % Separate data into three categories: Train, Validate and Testing
        TBL.Test        = TBL.All(idx.Test ,:);
        TBL.Val         = TBL.All(idx.Val  ,:);
        TBL.Train       = TBL.All(idx.Train,:);
        
        % MSE of the linear model compute by divide N-(Factor+1)
        mdlAll          = me.fitmodel(TBL.Train,ResponseVar,me.varList{:});
        
        % Coefficient at the min MSE
        [mdl,idxMSE,matMSE] = me.optmodel(TBL.Test,ResponseVar,mdlAll);
        
        % Collect the k-fold model
        me.mdl.(me.nameList{fold})= mdl;
        
        % Make the predicions
        Pred.Train  = me.predict( TBL.Train(:,idxTBL), fold );
        Pred.Val    = me.predict( TBL.Val  (:,idxTBL), fold );
        Pred.Test   = me.predict( TBL.Test (:,idxTBL), fold );
        Pred.All    = me.predict( TBL.All  (:,idxTBL), fold );
                
        % The prediction error
        Res.Train   = TBL.Train.(ResponseVar) - Pred.Train  ;
        Res.Val     = TBL.Val.  (ResponseVar) - Pred.Val    ;
        Res.Test    = TBL.Test. (ResponseVar) - Pred.Test   ;
        Res.All     = TBL.All.  (ResponseVar) - Pred.All    ;
        
        % The average value of every data set
        avg.Train   = mean(TBL.Train.(ResponseVar));
        avg.Val     = mean(TBL.Val.  (ResponseVar));
        avg.Test    = mean(TBL.Test. (ResponseVar));
        avg.All     = mean(TBL.All.  (ResponseVar));
        
        Diff.Train  = TBL.Train.(ResponseVar)  - avg.Train;
        Diff.Val    = TBL.Val.  (ResponseVar)  - avg.Val; 
        Diff.Test   = TBL.Test. (ResponseVar)  - avg.Test; 
        Diff.All    = TBL.All.  (ResponseVar)  - avg.All;
                
        if me.flagShow
            fprintf("Process of %s : \n", me.mdlTitle);
            strLength = strlength("Process of : ") + strlength(me.mdlTitle);
            fprintf("%s \n",me.underline(strLength));

            figure('Name',me.mdlTitle);

            % Plot the optimum lambda
            subplot(2,1,1);
            plot(me.Lambda(idxMSE),matMSE(idxMSE),'ro','MarkerSize',10);
            hold on;
            plot(me.Lambda,matMSE,'Color',[0.00,0.45,0.74]);
            text(me.Lambda(idxMSE),matMSE(idxMSE),strcat('$\lambda$ = ',num2str(me.Lambda(idxMSE))),'interpreter','latex','VerticalAlignment','bottom','HorizontalAlignment','left','Color','r','FontSize',10);
            hold off;
            title('Prediction Error','interpreter','latex')
            xlabel('Lambda, $\lambda$','interpreter','latex')
            ylabel('MSE','interpreter','latex')
            grid on;
            grid minor;
            axis tight;
            legend({'Optimal $\lambda$'},'Location','SE','interpreter','latex','FontSize',10);

            % Plot the fitting performance
            subplot(2,1,2);
            plot(TBL.Test.(ResponseVar),"o")
            hold on
            plot(Pred.Test,".")
            hold off
            title('Optimal Performance','interpreter','latex')
            xlabel('Sample No.','interpreter','latex')
            ylabel(ResponseVar,'interpreter','latex')
            grid on;
            grid minor;
            axis tight;
            legend({'Testing Data','Prediction'},'Location','SE','interpreter','latex','FontSize',10);
        end
                
        % Compute the paramber of the whole KFolds
        n.CrossVal         = n.CrossVal        + n.Test;
        sumCrossVal.AE     = sumCrossVal.AE    + sum ( abs ( Res.Test  )  );
        sumCrossVal.APE    = sumCrossVal.APE   + sum ( abs ( Res.Test  )./TBL.Test.(ResponseVar)  );
        sumCrossVal.BE     = sumCrossVal.BE    + sum (     ( Res.Test  )./TBL.Test.(ResponseVar)  );
        sumCrossVal.SE     = sumCrossVal.SE    + sum ( Res.Test.*Res.Test );
        CrossValData(idx.Test) = Pred.Test;
        
        % Compute the SE of the interval estimation 
        % and compute the mean of the error square
        MAE.All     = sum ( abs( Res.All )                 )/ n.All;  
        APE.All     = sum ( abs( Res.All )./TBL.All.(ResponseVar) )/ n.All;  
        MBE.All     = sum (    ( Res.All )./TBL.All.(ResponseVar) )/ n.All; 
        MSE.All     = sum (      Res.All.*Res.All     )/ n.All;  
        RMSE.All    = sqrt( MSE.All      );  
        HYP.All     = me.resHypothesis( Res.All );
        
        MAE.Train   = sum ( abs( Res.Train )                   )/ n.Train;  
        APE.Train   = sum ( abs( Res.Train )./TBL.Train.(ResponseVar) )/ n.Train;  
        MBE.Train   = sum (    ( Res.Train )./TBL.Train.(ResponseVar) )/ n.Train; 
        MSE.Train   = sum (      Res.Train.*Res.Train     )/ n.Train;  
        RMSE.Train  = sqrt( MSE.Train      );  
        HYP.Train   = me.resHypothesis( Res.Train );
        
        if( n.Val ~= 0 )
            MAE.Val     = sum ( abs( Res.Val )                     )/ n.Val;  
            APE.Val     = sum ( abs( Res.Val ) ./TBL.Val.(ResponseVar)    )/ n.Val;  
            MBE.Val     = sum (    ( Res.Val ) ./TBL.Val.(ResponseVar)    )/ n.Val; 
            MSE.Val     = sum (      Res.Val.*Res.Val         )/ n.Val; 
            RMSE.Val    = sqrt( MSE.Val      );  
            HYP.Val     = me.resHypothesis( Res.Val   );
        else
            MAE.Val     = missing;
            APE.Val     = missing;
            MBE.Val     = missing;
            MSE.Val     = missing;
            RMSE.Val    = missing;
            HYP.Val     = missing;
        end
                        
        MAE.Test         = sum ( abs( Res.Test )                    )/ n.Test;  
        APE.Test         = sum ( abs( Res.Test )./TBL.Test.(ResponseVar)   )/ n.Test;  
        MBE.Test         = sum (    ( Res.Test )./TBL.Test.(ResponseVar)   )/ n.Test; 
        MSE.Test         = sum ( Res.Test.*Res.Test            )/ n.Test;      
        RMSE.Test        = sqrt( MSE.Test      ); 
        HYP.Test         = me.resHypothesis( Res.Test  );
                
        MSEANOVA.All     = sum ( Res.All  .*Res.All   )/( n.All   - n.Factor - 1);      
        MSEANOVA.Train   = sum ( Res.Train.*Res.Train )/( n.Train - n.Factor - 1);      
        MSEANOVA.Val     = sum ( Res.Val  .*Res.Val   )/( n.Val   - n.Factor - 1);      
        MSEANOVA.Test    = sum ( Res.Test .*Res.Test  )/( n.Test  - n.Factor - 1);      
        RMSEANOVA.All    = sqrt( MSEANOVA.All   ); 
        RMSEANOVA.Train  = sqrt( MSEANOVA.Train );
        RMSEANOVA.Val    = sqrt( MSEANOVA.Val   );  
        RMSEANOVA.Test   = sqrt( MSEANOVA.Test  ); 
        
        % Compute the regression coefficient
        R2.Train     = 1 - sum( Res.Train.*Res.Train )  ...
                              / sum( Diff.Train    .*Diff.Train     );
        
        R2.Val       = 1 - sum( Res.Val  .*Res.Val   ) ...
                              / sum( Diff.Val      .*Diff.Val       );
        R2.Test      = 1 - sum( Res.Test .*Res.Test  ) ...
                              / sum( Diff.Test     .*Diff.Test      );
        R2.All       = 1 - sum( Res.All  .*Res.All   ) ...
                              / sum( Diff.All      .*Diff.All       );

        % Compute the adjusted R^2
        adjR2.Train  = 1 - (1-R2.Train)*( n.Train - 1)/( n.Train - n.Factor - 1);
        adjR2.Val    = 1 - (1-R2.Val  )*( n.Val   - 1)/( n.Val   - n.Factor - 1);
        adjR2.Test   = 1 - (1-R2.Test )*( n.Test  - 1)/( n.Test  - n.Factor - 1);
        adjR2.All    = 1 - (1-R2.All  )*( n.All   - 1)/( n.All   - n.Factor - 1); 
                
        % Summary the model performance
        varNames   = {'R^2','adj.R^2'};
        rowNames   = {'Train','Validate','Test','All'};
        R2Data     = [R2.Train   ;R2.Val   ;R2.Test   ;R2.All   ];
        adjR2Data  = [adjR2.Train;adjR2.Val;adjR2.Test;adjR2.All];
        me.R2TBL.( me.nameList{fold} )   = table(R2Data, adjR2Data , 'VariableNames', varNames, 'RowNames', rowNames );
        
        varNames   = {'Train','Validate','Test','All'};
        rowNames   = {'MAE','APE','MBE','MSE','RMSE','MSEANOVA','RMSEANOVA'};
        Eva.Train  = [ MAE.Train;APE.Train;MBE.Train;MSE.Train;RMSE.Train;MSEANOVA.Train;RMSEANOVA.Train ];
        Eva.Val    = [ MAE.Val  ;APE.Val  ;MBE.Val  ;MSE.Val  ;RMSE.Val  ;MSEANOVA.Val  ;RMSEANOVA.Val   ];
        Eva.Test   = [ MAE.Test ;APE.Test ;MBE.Test ;MSE.Test ;RMSE.Test ;MSEANOVA.Test ;RMSEANOVA.Test  ];
        Eva.All    = [ MAE.All  ;APE.All  ;MBE.All  ;MSE.All  ;RMSE.All  ;MSEANOVA.All  ;RMSEANOVA.All   ];
        me.EvaTBL.( me.nameList{fold} )  = table(Eva.Train ,Eva.Val ,Eva.Test ,Eva.All ,'VariableNames', varNames, 'RowNames', rowNames );
    
        varNames   = {'Train','Validate','Test','All'};
        me.Residuals.( me.nameList{fold} ).Data  = { Res.Train, Res.Val, Res.Test, Res.All };
        me.Residuals.( me.nameList{fold} ).Name  = varNames;
        
        me.ResidualsTest.( me.nameList{fold} ).Train = HYP.Train;
        me.ResidualsTest.( me.nameList{fold} ).Val   = HYP.Val;
        me.ResidualsTest.( me.nameList{fold} ).Test  = HYP.Test;
        me.ResidualsTest.( me.nameList{fold} ).All   = HYP.All;
                
        fprintf("Summary of %s : \n", me.mdlTitle);
        strLength = strlength("Summary of : ") + strlength(me.mdlTitle);
        fprintf("%s \n",me.underline(strLength));
        fprintf("R^2 : %s \n", me.nameList{fold} );
        disp(me.R2TBL.(me.nameList{fold}) );
        %fprintf("ANOVA : \n")
        %disp(mdl.Coefficients);
        %disp(anova(mdl,'summary'));

    end

    if me.chkCV == false || strcmp(lower(me.c.Type),lower('Holdout'))
        return;
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The summary of K-Fold Statistic %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The average error based on the cross validation
    MAE.CrossVal        = sumCrossVal.AE  /  n.CrossVal;
    APE.CrossVal        = sumCrossVal.APE /  n.CrossVal;
    MBE.CrossVal        = sumCrossVal.BE  /  n.CrossVal;
    MSE.CrossVal        = sumCrossVal.SE  /  n.CrossVal;
    RMSE.CrossVal       = sqrt( MSE.CrossVal );
    MSEANOVA.CrossVal   = sumCrossVal.SE  /( n.CrossVal-n.Factor-1 );
    RMSEANOVA.CrossVal  = sqrt( MSEANOVA.CrossVal  );

    varNames   =  {'CrossVal'};
    rowNames   =  {'MAE','APE','MBE','MSE','RMSE','MSEANOVA','RMSEANOVA'};
    Eva.CrossVal    = [ MAE.CrossVal;APE.CrossVal;MBE.CrossVal;MSE.CrossVal;RMSE.CrossVal;MSEANOVA.CrossVal;RMSEANOVA.CrossVal ];
    me.EvaTBL.( me.nameList{end} )   = table(Eva.CrossVal ,'VariableNames',varNames,'RowNames',rowNames );
 
    % The R^2 of the cross validation
    Res.CrossVal     = TBL.All.(ResponseVar) - CrossValData;
    R2.CrossVal      = 1 - sum( Res.CrossVal.* Res.CrossVal ) ...
                              / sum( Diff.All.*Diff.All );
    adjR2.CrossVal   = 1 - (1-R2.CrossVal  )*( n.CrossVal - 1)/( n.CrossVal - n.Factor - 1);
       
    varNames              = {'R^2','adj.R^2'};
    rowNames              = {'CrossVal'};
    R2Data                = [    R2.CrossVal ];
    adjR2Data             = [ adjR2.CrossVal ];
    me.R2TBL.( me.nameList{end}  ) = table( R2Data,adjR2Data,'VariableNames',varNames,'RowNames',rowNames );
    
    varNames              = {'CrossVal'};
    me.Residuals.( me.nameList{end} ).Data   = Res.CrossVal;
    me.Residuals.( me.nameList{end} ).Name   = varNames;

    fprintf("Summary of %s : \n",me.mdlTitle);
    fprintf("================================== \n");
    fprintf("The validation approach : %s %i \n",me.c.Type, me.c.NumTestSets);
    disp(me.R2TBL.   ( me.nameList{end} ));
    disp(me.EvaTBL.  ( me.nameList{end} ));

    % The residual diagnostic
    me.resDiagnostic( Res.CrossVal  , TBL.All.(ResponseVar), CrossValData  , RMSEANOVA.CrossVal  , strcat("The residual of ",me.mdlTitle) );
    
    % Perform the normaility testing of the residual
    HYP.CrossVal = me.resHypothesis( Res.CrossVal );
    
    % Return the results of normality test
    me.ResidualsTest.CrossVal = HYP.CrossVal;
   
end