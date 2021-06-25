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

%%%%%%%%%%%%%%%%%%%%%%%
% Residual Diagnostic %
%%%%%%%%%%%%%%%%%%%%%%%

function resDiagnostic( me, res, ActData, PredData, RMSE, strTitle )

    % Create new figure supporting the residual diagonostic
    figRes = figure('Name',"Residual Diagnostic");
    
    % Preparing for the interval estimation
    xDomain = linspace(min(PredData),max(PredData),100);
   
    % Plot the 95% confident level of the predicted data
    subplot(2,2,1)
    plot(PredData,PredData,'r-','LineWidth',1)
    hold on
    plot(xDomain,xDomain+norminv(.025)*RMSE,'r--','LineWidth',0.5)
    plot(xDomain,xDomain+norminv(.975)*RMSE,'r--','LineWidth',0.5)
    plot(PredData,ActData,'o','Color',[0.00,0.45,0.74])
    hold off
    set(gca,'FontName','Times New Roman');
    axis tight
    grid on
    grid minor
    title('Performance of the model','interpreter','latex','fontsize',12)
    ylabel('Actual HHV value','interpreter','latex','fontsize',12)
    xlabel('Predicted HHV value','interpreter','latex','fontsize',12)
    legend({'95% Confident Level'},'Location','northwest')

    % Plot the model residual of each data
    subplot(2,2,2)
    plot(res,'o')
    hold on
    yline(0,'r','LineWidth',2)
    hold off
    set(gca,'FontName','Times New Roman');
    axis tight
    axis tight
    grid on
    grid minor
    title('Standardized Residuals','interpreter','latex','fontsize',12)
    xlabel('No. of data','interpreter','latex','fontsize',12)
    ylabel('Residual value','interpreter','latex','fontsize',12)

    % Plot the histogram of the model residual
    subplot(2,2,3)
    y = linspace(min(res),max(res),100);
    mu = mean(res);
    sigma = std(res);
    f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
    plot(y,f,'r-.','LineWidth',1.)
    hold on
    histogram(res,'Normalization','pdf','FaceColor',[0.00,0.45,0.74])
    title('The distribution of the standardized residuals','interpreter','latex','fontsize',12)
    set(gca,'FontName','Times New Roman');
    xlabel('Residuals value','interpreter','latex','fontsize',12)
    ylabel('P.D.F.','interpreter','latex','fontsize',12)
    legend('Normal P.D.F.','interpreter','latex','Location','best');
    hold off
    axis tight
    grid on
    grid minor

    % Create the QQ Plot of the model residual
    subplot(2,2,4)
    qqplot(res)
    axis tight
    set(gca,'FontName','Times New Roman');
    title('QQ Plot of the standardized residuals','interpreter','latex','fontsize',12)
    xlabel('Standard normal quantitle','interpreter','latex','fontsize',12)
    ylabel('Quantiles','interpreter','latex','fontsize',12)
    grid on
    grid minor
    
    % Set the property of the figure
    set(figRes,'units','normalized','outerposition',[0. 0. 1 1]);
    if (nargin>4)
        set(figRes,'Name' ,strTitle);
    end
    
end