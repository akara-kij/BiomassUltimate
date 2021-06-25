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

function mdlDiagnostic(me,mdl)

    % Exhibit the process of residual diagnostic
    fprintf("Process of %s : \n", me.mdlTitle);
    strLength = strlength("Process of : ") + strlength(me.mdlTitle);
    fprintf("%s \n",me.underline(strLength));
    
    % Routine parameter to compute the residual of the test data
    n.All           = size ( me.dataTBL,1);
    n.Factor        = size ( me.dataTBL,2)-1; % [ FC VM A HHV ]
    n.Test          = round( me.pFactor(1)*n.All );
    NData           = [1:n.All];
    idxTBL          = me.dataTBL.Properties.VariableNames ~= me.ResponseVarName ;
    idx.Test        = datasample( me.sRand,NData ,n.Test,'Replace',false );
    
    % Compute the residual of the predicted data
    TBL.Test        = me.dataTBL(idx.Test ,:);
    Pred.Test       = me.predict( TBL.Test (:,idxTBL) );
    Res.Test        = TBL.Test. (me.ResponseVarName) - Pred.Test   ;
    MSEANOVA.Test   = sum ( Res.Test .*Res.Test  )/( n.Test  - n.Factor - 1);      
    RMSEANOVA.Test  = sqrt( MSEANOVA.Test  ); 

    % Present Residual plot
    resDiagnostic( Res.Test  , TBL.Test.(me.ResponseVarName), Pred.Test  , RMSEANOVA.Test  , strcat("The cross validation residual of ",me.mdlTitle) );

end

