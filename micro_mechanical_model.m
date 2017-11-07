% Comparison to Micromechanical model
clear;clc;clf;

% Load the data
[~,~,Data] = xlsread('CompleteDataNoMineralComps.csv');
coarseAggType = Data(:,27);
sieveVals = xlsread('Sievepercents.csv');

% Define aggregate CTEs (from paper)
QuartzCTE = 12;
DolomiteCTE = 8.5;
BasaltCTE = 7;

% Sieve Size (mm)
sieve_size = [37.5, 25, 19, 12,5, 9.5, 4.75, 2.36];


%%
for i = 2:length(Data(:,1))
    
    disp(i)
    
    % reset the counter
    check = 0;
    
    % Predefine values that change throughout iteration
    basicCTEval = 0;
    mixSieveVals = zeros(1,length(sieve_size));
    
    % Get the sieve percents from the sieve values for the specific mix
    % number
    if strcmp(coarseAggType(i),'GG1') == 1
        basicCTEval = QuartzCTE;
        mixSieveVals = sieveVals(1,1:end);
    elseif strcmp(coarseAggType(i),'GG2') == 1
        basicCTEval = DolomiteCTE;
        mixSieveVals = sieveVals(2,1:end);
    elseif strcmp(coarseAggType(i),'GG3') == 1
        basicCTEval = DolomiteCTE;
        mixSieveVals = sieveVals(3,1:end);
    elseif strcmp(coarseAggType(i),'GG4') == 1
        basicCTEval = QuartzCTE;
        mixSieveVals = sieveVals(4,1:end);
    elseif strcmp(coarseAggType(i),'GG5') == 1
        basicCTEval = DolomiteCTE;
        mixSieveVals = sieveVals(5,1:end);
    elseif strcmp(coarseAggType(i),'GG6') == 1
        basicCTEval = DolomiteCTE;
        mixSieveVals = sieveVals(6,1:end);
    elseif strcmp(coarseAggType(i),'CS1') == 1
        basicCTEval = DolomiteCTE;
        mixSieveVals = sieveVals(7,1:end);
    elseif strcmp(coarseAggType(i),'CS2') == 1
        basicCTEval = QuartzCTE;
        mixSieveVals = sieveVals(8,1:end);
    elseif strcmp(coarseAggType(i),'CS3') == 1
        basicCTEval = DolomiteCTE;
        mixSieveVals = sieveVals(9,1:end);
    elseif strcmp(coarseAggType(i),'CS4') == 1
        basicCTEval = DolomiteCTE;
        mixSieveVals = sieveVals(10,1:end);
    elseif strcmp(coarseAggType(i),'CS5') == 1
        basicCTEval = DolomiteCTE;
        mixSieveVals = sieveVals(11,1:end);
    elseif strcmp(coarseAggType(i),'CS6') == 1
        basicCTEval = QuartzCTE;
        mixSieveVals = sieveVals(12,1:end);
    elseif strcmp(coarseAggType(i),'CS7') == 1
        basicCTEval = BasaltCTE;
        mixSieveVals = sieveVals(13,1:end);
    elseif strcmp(coarseAggType(i),'CS8') == 1
        basicCTEval = DolomiteCTE;
        mixSieveVals = sieveVals(14,1:end);
    elseif strcmp(coarseAggType(i),'CS9') == 1
        basicCTEval = QuartzCTE;
        mixSieveVals = sieveVals(15,1:end);
    end
    
    % Set a counter to be used in while loop
    counter = 0;
    % Initialize a value of alpha 0
    alpha0 = 0;
    while check == 0
        
        % only set value of alpha 0 if first iteration
        if counter == 0
            alpha0_old = 0;
        end
        
        % SUMMATION OVER COARSE AGGREGATE SIZE. Subtract 1 because of
        % indexing.
        for m = 1:(length(sieveVals(1,:))-1)
            sieveSize = sieve_size(m);
            % What is the difference between the CTE for alpha(a_ij) and
            % alpha(a_(i+1)j?
            alpha0 = alpha0_old + 1/2*(basicCTEval + basicCTEval)*(mixSieveVals(7-m) - mixSieveVals(8-m));
        end 
        
        % Calculate the error
        error = alpha0 - alpha0_old;
        
        disp('the error is')
        disp(alpha0)
        disp(alpha0_old)
        disp('----------')
        
        % Set the current value of alpha0 to the old value
        alpha0_old = alpha0;
        
        % 0.3 because values were are using have been multiplied by 1e6
        if error <= 0.3
            check = 1;
        end
        
        % iterate the counter
        counter = counter + 1;
    end
    disp('tolerance reached')
end