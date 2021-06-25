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

function me = fit(me,varargin)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Routine to fit model by the neural network %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           
    % The routine structure variable
    n           = struct('Train',0 , 'Val'  ,0 , 'Test',0 , 'CrossVal',10 , 'Factor'  ,0, 'All',0);
    sumCrossVal = struct('AE'   ,[], 'APE'  ,[], 'BE'  ,[], 'SE'      ,[] );
    idx         = struct('All'  ,[], 'Train',[], 'Val' ,[], 'Test'    ,[] , 'CrossVal',[] );
    Eva         = idx;  % Evaluation results
    R2          = Eva;  % The R Square
    adjR2       = Eva;  % The adjusted R Square
    avg         = Eva;  % The average value
    Diff        = Eva;  % The sum of total error
    Res         = Eva;  % The model resiudals 
    MAE         = Eva;  % The mean absolute value
    APE         = Eva;  % The mean absolute proportional error
    MBE         = Eva;  % The mean bias error
    MSE         = Eva;  % The mean square error
    RMSE        = Eva;  % The root of mean square error
    MSEANOVA    = Eva;  % The mean square error of ANOVA
    RMSEANOVA   = Eva;  % The root mean square error of ANOVA
    HYP         = Eva;  % The normality test of the residuals
    
    % Prepare the array of data
    idxTBL      = me.dataTBL.Properties.VariableNames ~= me.ResponseVarName ;
    XData       = me.dataTBL{:,idxTBL};
    YData       = me.dataTBL{:,me.ResponseVarName};
    
    % Filter the data to the focus one
    if isnumeric(me.dataID) && isvector(me.dataID)
        XData   = XData(me.dataID,:);
        YData   = YData(me.dataID,:);
    end
    
    % Transpose column data into a row matrix
    XData       = XData';
    YData       = YData';

    % The number of observation
    n.All       = size(YData  ,2);
    n.Factor    = size(XData  ,1);
    n.Layers    = length(me.nNeuron);
    me.chkCV    = false;
        
    % Extract the model Name-Value parameters
    me.extractVargin(varargin{:});

    % Memory Allocation
    n.CrossVal        = 0;
    sumCrossVal.AE    = 0.;
    sumCrossVal.APE   = 0.;
    sumCrossVal.BE    = 0.;
    sumCrossVal.SE    = 0.;
    CrossValData      = YData;
    
    % Perform K-fold evaluation
    for fold=1:me.KFolds

        % Create the model of the neural network
        %mdl = me.fitmodel(TBL.Train,me.ResponseVarName,me.varList{:});
        mdl = me.fitmodel(XData,YData);
        
        % Assign transfer function of the hidden layer
        % 'tansig' : Hyperbolic tangent sigmoid transfer function
        % 'logsig' : Log-sigmoid transfer function
        % 'radbas' : Radial Basis Neural Networks
        % 'purelin': Linear fucntion
        if ~ isempty(me.transFcn)
            for i = 1: n.Layers
                mdl.layers{i}.transferFcn = me.transFcn{i};
            end
        end
        
        % Setting the performance function
        % For a list of all performance functions type: help nnperformance
        %   'mse'       : Mean Squared Error
        %   'mae'       : Mean Absolute Error , Work only percepton not N.N.!!!
        mdl.performFcn = 'mse'; 

        % Choose Plot Functions
        % For a list of all available plot button : specify to enable the
        % plot function
        mdl.plotFcns   = {'plotperform'   ,'plottrainstate','ploterrhist', ...
                          'plotregression', 'plotfit'                    };

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
                    idx.Train         = training(me.c);
                    idx.Test          = test(me.c);
               otherwise

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Network parameters : divideFcn %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % divideind   : Divide the data by index
                    % dividetrain : Assign all targets to training set
                    idx.Train         = training(me.c,fold);
                    idx.Test          = test(me.c,fold);
           end
           % Split the Data for based in assigned index
           NData = [1:n.All];
           me.divideFcn = 'divideind';
           mdl.divideFcn              = me.divideFcn;
           mdl.divideParam.trainInd   = NData(idx.Train);
           mdl.divideParam.valInd     = [];
           mdl.divideParam.testInd    = NData(idx.Test);
        else 
           %
           % Hold out & Resubstitution
           % 
           % Split the Data for Training, Validation, Testing
           pTestData                  = me.pFactor(1);
           pValidateData              = me.pFactor(2);
           pTrainData                 = 1-sum( me.pFactor );
           mdl.divideFcn              = me.divideFcn; 
           mdl.divideMode             = 'sample'; 
           mdl.divideParam.trainRatio = pTrainData;
           mdl.divideParam.valRatio   = pValidateData;
           mdl.divideParam.testRatio  = pTestData;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Neural Network Computation %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Set Input and Output Pre/Post-Processing Functions
        %   'mapminmax'  : Normalize inputs/targets to fall in the range [−1, 1]
        %   'mapstd'     : Normalize inputs/targets to have zero mean and unity variance
        %   'processpca' : Extract principal components from the input vector
        %   'fixunknowns': Process unknown inputs
        %   'removeconstantrows':removeconstantrows
        mdl.trainParam.epochs  =  2e3;
        mdl.input.processFcns  = me.inputProcessFcns;
        mdl.output.processFcns = me.outputProcessFcns;

        % Train the network basing on the train data
        [mdl, tr]   = me.train(mdl,XData,YData);

        % Test the Network : Use matlab function
        PredData    = mdl(XData,me.varList{:});
        
        errAll      = gsubtract(YData,PredData);
        mMSE        = mse(mdl,YData,PredData);
        mMAE        = mae(errAll);
        mRMSE       = sqrt( mMSE );

        % Reload the Training, Validation and Test data
        trainData   = YData .* tr.trainMask{1};
        valData     = YData .* tr.valMask{1};
        testData    = YData .* tr.testMask{1};
 
        % Extract the data in every process
        idx.Train   = ~isnan(trainData) ;
        idx.Val     = ~isnan(valData  );
        idx.Test    = ~isnan(testData );
        
        % The number of each data set
        n.Train     = nnz( idx.Train );
        n.Val       = nnz( idx.Val   );
        n.Test      = nnz( idx.Test  );

        % The prediction error
        Res.Train     = trainData(idx.Train) - PredData(idx.Train);
        Res.Val       = valData  (idx.Val  ) - PredData(idx.Val  );
        Res.Test      = testData (idx.Test ) - PredData(idx.Test );
        Res.All       = PredData             - YData;
                 
        % Comfirm the result of the MSE function , Inspect the formula
        calMAE        = mean( abs( Res.All ) );
        mMBE          = mean(    ( Res.All )./YData );
        mIMMSE        = immse( YData,PredData );% Image processing toolbox
        
        % Compute the paramber of the whole KFolds
        n.CrossVal             = n.CrossVal     + n.Test;
        sumCrossVal.AE         = sumCrossVal.AE + sum ( abs ( Res.Test )   );
        sumCrossVal.APE        = sumCrossVal.APE+ sum ( abs ( Res.Test ) ./ testData(idx.Test )  );
        sumCrossVal.BE         = sumCrossVal.BE + sum (     ( Res.Test ) ./ testData(idx.Test )  );
        sumCrossVal.SE         = sumCrossVal.SE + sum ( Res.Test.*Res.Test );
        CrossValData(idx.Test) = PredData(idx.Test);
                              
        % Compute the SE of the interval estimation 
        % and compute the mean of the error square
        MAE.All         = sum ( abs( Res.All )        )/ n.All;  
        APE.All         = sum ( abs( Res.All )./YData )/ n.All;  
        MBE.All         = sum (    ( Res.All )./YData )/ n.All; 
        MSE.All         = sum (      Res.All.*Res.All )/ n.All;  
        RMSE.All        = sqrt( MSE.All );   
        HYP.All         = me.resHypothesis( Res.All   );
        
        MAE.Train       = sum ( abs( Res.Train )                       )/ n.Train;  
        APE.Train       = sum ( abs( Res.Train )./trainData(idx.Train) )/ n.Train;  
        MBE.Train       = sum (    ( Res.Train )./trainData(idx.Train) )/ n.Train; 
        MSE.Train       = sum (      Res.Train.*Res.Train              )/ n.Train;  
        RMSE.Train      = sqrt( MSE.Train );  
        HYP.Train       = me.resHypothesis( Res.Train );
        
        if( n.Val ~= 0 )
            MAE.Val     = sum ( abs( Res.Val )                        )/ n.Val;  
            APE.Val     = sum ( abs( Res.Val )  ./valData  (idx.Val ) )/ n.Val;  
            MBE.Val     = sum (    ( Res.Val )  ./valData  (idx.Val ) )/ n.Val; 
            MSE.Val     = sum (      Res.Val.*Res.Val                 )/ n.Val; 
            RMSE.Val    = sqrt( MSE.Val   );
            HYP.Val     = me.resHypothesis( Res.Val   );
        else
            MAE.Val     = missing;
            APE.Val     = missing;
            MBE.Val     = missing;
            MSE.Val     = missing;
            RMSE.Val    = missing;
            HYP.Val     = missing;
        end
        
        MAE.Test        = sum ( abs( Res.Test )                       )/ n.Test;  
        APE.Test        = sum ( abs( Res.Test )./testData (idx.Test ) )/ n.Test;  
        MBE.Test        = sum (    ( Res.Test )./testData (idx.Test ) )/ n.Test; 
        MSE.Test        = sum ( Res.Test.*Res.Test )/n.Test;      
        RMSE.Test       = sqrt( MSE.Test  ); 
        HYP.Test        = me.resHypothesis( Res.Test  );
                
        MSEANOVA.All    = sum ( Res.All  .*Res.All   )/( n.All   - n.Factor - 1);      
        MSEANOVA.Train  = sum ( Res.Train.*Res.Train )/( n.Train - n.Factor - 1);      
        MSEANOVA.Val    = sum ( Res.Val  .*Res.Val   )/( n.Val   - n.Factor - 1);      
        MSEANOVA.Test   = sum ( Res.Test .*Res.Test  )/( n.Test  - n.Factor - 1);      
        RMSEANOVA.All   = sqrt( MSEANOVA.All   ); 
        RMSEANOVA.Train = sqrt( MSEANOVA.Train );
        RMSEANOVA.Val   = sqrt( MSEANOVA.Val   );  
        RMSEANOVA.Test  = sqrt( MSEANOVA.Test  ); 

        % Compute the performance in every process
        Performance.Train = perform(mdl,trainData,PredData);
        Performance.Val   = perform(mdl,valData  ,PredData);
        Performance.Test  = perform(mdl,testData ,PredData);
        Performance.All   = perform(mdl,YData    ,PredData);

        % The average value of every data set
        avg.Train       = mean(trainData(idx.Train));
        avg.Val         = mean(valData(idx.Val));
        avg.Test        = mean(testData(idx.Test));
        avg.All         = mean(YData);

        % Correct the R^2
        Diff.Train      = trainData(idx.Train)- avg.Train;
        Diff.Val        = valData(idx.Val)    - avg.Val; 
        Diff.Test       = testData(idx.Test)  - avg.Test; 
        Diff.All        = YData               - avg.All;

        R2.Train        = 1 - sum( Res.Train .*Res.Train  ) ...
                            / sum( Diff.Train.*Diff.Train );
        R2.Val          = 1 - sum( Res.Val   .*Res.Val    ) ...
                            / sum( Diff.Val  .*Diff.Val   );
        R2.Test         = 1 - sum( Res.Test  .*Res.Test   ) ...
                            / sum( Diff.Test .*Diff.Test  );
        R2.All          = 1 - sum( Res.All   .*Res.All    ) ...
                            / sum( Diff.All  .*Diff.All   );

        % Compute the adjusted R^2
        adjR2.Train = 1 - (1-R2.Train)*( n.Train - 1)/( n.Train - n.Factor - 1);
        adjR2.Val   = 1 - (1-R2.Val  )*( n.Val   - 1)/( n.Val   - n.Factor - 1);
        adjR2.Test  = 1 - (1-R2.Test )*( n.Test  - 1)/( n.Test  - n.Factor - 1);
        adjR2.All   = 1 - (1-R2.All  )*( n.All   - 1)/( n.All   - n.Factor - 1); 

        % Summary the model performance
        varNames    = {'R^2'  ,'adj.R^2'};
        rowNames    = {'Train','Validate','Test','All'};
        R2Data      = [R2.Train   ;R2.Val   ;R2.Test   ;R2.All   ];
        adjR2Data   = [adjR2.Train;adjR2.Val;adjR2.Test;adjR2.All];
        me.R2TBL.( me.nameList{fold} ) = table(R2Data,adjR2Data ,'VariableNames',varNames,'RowNames',rowNames );
       
        varNames    = {'Train','Validate','Test','All'};
        rowNames    = {'MAE','APE','MBE','MSE','RMSE','MSEANOVA','RMSEANOVA'};
        Eva.Train   = [ MAE.Train;APE.Train;MBE.Train;MSE.Train;RMSE.Train;MSEANOVA.Train;RMSEANOVA.Train ];
        Eva.Val     = [ MAE.Val  ;APE.Val  ;MBE.Val  ;MSE.Val  ;RMSE.Val  ;MSEANOVA.Val  ;RMSEANOVA.Val   ];
        Eva.Test    = [ MAE.Test ;APE.Test ;MBE.Test ;MSE.Test ;RMSE.Test ;MSEANOVA.Test ;RMSEANOVA.Test  ];
        Eva.All     = [ MAE.All  ;APE.All  ;MBE.All  ;MSE.All  ;RMSE.All  ;MSEANOVA.All  ;RMSEANOVA.All   ];
        me.EvaTBL.( me.nameList{fold} ) = table(Eva.Train ,Eva.Val ,Eva.Test ,Eva.All ,'VariableNames',varNames,'RowNames',rowNames );
        
        varNames    = {'Train','Validate','Test','All'};
        me.Residuals.( me.nameList{fold} ).Data  = { Res.Train', Res.Val', Res.Test', Res.All' };
        me.Residuals.( me.nameList{fold} ).Name  = varNames;
        
        me.ResidualsTest.( me.nameList{fold} ).Train = HYP.Train;
        me.ResidualsTest.( me.nameList{fold} ).Val   = HYP.Val;
        me.ResidualsTest.( me.nameList{fold} ).Test  = HYP.Test;
        me.ResidualsTest.( me.nameList{fold} ).All   = HYP.All;
        
        % Collect the evaluation model
        me.mdl.( me.nameList{fold} ) = mdl;
        me.tr. ( me.nameList{fold} ) = tr;  
        
        fprintf("Summary of the model performance : \n");
        fprintf("================================== \n");
        fprintf("The model algorithm : %s \n",me.trainFcn);
        switch me.divideFcn
            case lower('dividetrain')
                strTitle = 'Resubstituion of all data';
            otherwise
                strTitle = ['pTest = ',num2str(me.pFactor(1)),', pValidate = ',num2str(me.pFactor(2))];
        end
        fprintf("The validation approach : %s , %s \n", me.nameList{fold}, strTitle );
        
        disp(me.R2TBL.( me.nameList{fold} ));
        disp(me.EvaTBL.( me.nameList{fold} ));

        % Display the model performance
        if (me.flagShow == false) 
            continue; 
        end

        % View the neural network
        view(mdl); 

        % The error histogram
        figErr      = figure;
        ploterrhist(Res.Train,'Train',Res.Val,'Validate',Res.Test,'Test')

        % The state of training process
        figState    = figure;
        plottrainstate(tr)

        % The regression of each proces
        figRegress  = figure;
        plotregression(trainData(idx.Train),PredData(idx.Train),'Train'   ,...
                       valData  (idx.Val  ),PredData(idx.Val  ),'Validate',...
                       testData (idx.Test ),PredData(idx.Test ),'Test'    ,...
                       YData               ,PredData           ,'All'     );

        % The ANN performance           
        figPref     = figure;
        plotperform(tr)

        % The function fit
        if (size(XData,1) == 1)
            figure;
            plotfit(mdl,XData,YData); 
        end

        % Configure overview of figure
        set(figRegress,'units','normalized','outerposition',[0.0 0.0  .5 1.00]);
        set(figErr    ,'units','normalized','outerposition',[0.5 0.66 .5  .34]);
        set(figState  ,'units','normalized','outerposition',[0.5 0.33 .5  .33]);
        set(figPref   ,'units','normalized','outerposition',[0.5 0.0  .5  .33]);
        set(groot     ,'CurrentFigure',figRegress);

        % Inspect the residual errors in details
        if ( nnz(idx.Train) ~=0 )  
            me.resDiagnostic( Res.Train, trainData(idx.Train), PredData(idx.Train), RMSEANOVA.Train, "The residual of the training data" );
        end
        if ( nnz(idx.Val) ~=0   )  
            me.resDiagnostic( Res.Val  , valData(idx.Val)    , PredData(idx.Val  ), RMSEANOVA.Val  , "The residual of the validate data" );
        end
        if ( nnz(idx.Test) ~=0  )  
            me.resDiagnostic( Res.Test , testData(idx.Test)  , PredData(idx.Test ), RMSEANOVA.Test , "The residual of the testing data"  );
        end
        me.resDiagnostic( Res.All, YData, PredData, RMSEANOVA.All, "The residual of all data" );
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The summary of K-Fold Statistic %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Bypass the processs
    if me.chkCV == false || strcmp(lower(me.c.Type),lower('Holdout'))
        return;
    end
    
    % The average error based on the cross validation
    MAE.CrossVal        = sumCrossVal.AE  /  n.CrossVal;
    APE.CrossVal        = sumCrossVal.APE /  n.CrossVal;
    MBE.CrossVal        = sumCrossVal.BE  /  n.CrossVal;
    MSE.CrossVal        = sumCrossVal.SE  /  n.CrossVal;
    RMSE.CrossVal       = sqrt( MSE.CrossVal );
    MSEANOVA.CrossVal   = sumCrossVal.SE  /( n.CrossVal - n.Factor-1 );
    RMSEANOVA.CrossVal  = sqrt( MSEANOVA.CrossVal  );
    
    % The R^2 of the cross validation
    Res.CrossVal        = YData - CrossValData;
    R2.CrossVal         = 1 - sum( Res.CrossVal.*Res.CrossVal ) ...
                            / sum( Diff.All.*Diff.All );
    adjR2.CrossVal      = 1 - (1-R2.CrossVal  )*( n.CrossVal - 1)/( n.CrossVal - n.Factor - 1 );
    
    % Report the cross validation parameter 
    varNames            = {'R^2','adj.R^2'};
    rowNames            = {'CrossVal'};
    R2Data              = [ R2.CrossVal    ];
    adjR2Data           = [ adjR2.CrossVal ];
    me.R2TBL.( me.nameList{end} ) = table(R2Data,adjR2Data ,'VariableNames',varNames,'RowNames',rowNames );

    varNames            = {'CrossVal'};
    rowNames            = {'MAE','APE','MBE','MSE','RMSE','MSEANOVA','RMSEANOVA'};
    Eva.CrossVal        = [MAE.CrossVal;APE.CrossVal;MBE.CrossVal;MSE.CrossVal;RMSE.CrossVal;MSEANOVA.CrossVal;RMSEANOVA.CrossVal];
    me.EvaTBL.( me.nameList{end} ) = table(Eva.CrossVal,'VariableNames',varNames,'RowNames',rowNames );

    varNames            = {'CrossVal'};
    me.Residuals.( me.nameList{end} ).Data    = Res.CrossVal';
    me.Residuals.( me.nameList{end} ).Name    = varNames;
    
    fprintf("Summary of the model performance : \n");
    fprintf("================================== \n");
    fprintf("The model algorithm : %s \n",me.trainFcn);
    fprintf("The validation approach : %s %i \n",me.c.Type, me.c.NumTestSets);
    disp(me.R2TBL. ( me.nameList{end} ));
    disp(me.EvaTBL.( me.nameList{end} ));
    
    % The residual diagnostic
    me.resDiagnostic( Res.CrossVal, YData, PredData, RMSEANOVA.All, strcat("The cross validation residual of ",me.mdlTitle) );
    
    % Perform the normaility testing of the residual
    HYP.CrossVal = me.resHypothesis( Res.CrossVal );
    
    % Return the results of normality test
    me.ResidualsTest.CrossVal = HYP.CrossVal;
    
end