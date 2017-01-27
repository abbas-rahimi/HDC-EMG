	% This program implements the use of hyperdimensional (HD) computing to 
	% classify electromyography (EMG) signals for hand gesture recognition. 
	% Copyright (C) 2016 Abbas Rahimi (e-mail:abbas@eecs.berkeley.edu).

	% This program is free software: you can redistribute it and/or modify
	% it under the terms of the GNU General Public License as published by
	% the Free Software Foundation, either version 3 of the License, or
	% (at your option) any later version.

	% This program is distributed in the hope that it will be useful,
	% but WITHOUT ANY WARRANTY; without even the implied warranty of
	% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	% GNU General Public License for more details.

	% You should have received a copy of the GNU General Public License
	% along with this program.  If not, see <http://www.gnu.org/licenses/>.

function message = ICRC
  assignin('base','lookupItemMemeory', @lookupItemMemeory);
  assignin('base','initItemMemories', @initItemMemories);
  assignin('base','genRandomHV', @genRandomHV);
  assignin('base','cosAngle', @cosAngle);
  assignin('base','normDot', @normDot);
  assignin('base','computeSumHV', @computeSumHV);
  assignin('base','computeNgram', @computeNgram);
  assignin('base','hdctrain', @hdctrain); 
  assignin('base','downSampling', @downSampling);
  assignin('base','hdcpredict', @hdcpredict);
  assignin('base','genTrainData', @genTrainData);
  assignin('base','test_slicing', @test_slicing);
  assignin('base','findbestN', @findbestN);
  message='Importing all HD computing functions to workspace is done';
end

function [L_SAMPL_DATA, SAMPL_DATA] = genTrainData (data, labels, trainingFrac, order)
%
% DESCRIPTION   : generate a dataset useful for training using a fraction of input data 
%
% INPUTS:
%   data        : input data
%   labels      : input labels
%   trainingFrac: the fraction of data we shouls use to output a training dataset
%   order       : whether preserve the order of inputs (inorder) or randomly select
%   donwSampRate: the rate or stride of downsampling
% OUTPUTS:
%   SAMPL_DATA  : dataset for training
%   L_SAMPL_DATA: corresponding labels
%    

	rng('default');
    rng(1);
    L1 = find (labels == 1);
    L2 = find (labels == 2);
    L3 = find (labels == 3);
    L4 = find (labels == 4);
    L5 = find (labels == 5);
	L6 = find (labels == 6);
	L7 = find (labels == 7);
   
    L1 = L1 (1 : floor(length(L1) * trainingFrac));
    L2 = L2 (1 : floor(length(L2) * trainingFrac));
    L3 = L3 (1 : floor(length(L3) * trainingFrac));
    L4 = L4 (1 : floor(length(L4) * trainingFrac));
    L5 = L5 (1 : floor(length(L5) * trainingFrac));
	L6 = L6 (1 : floor(length(L6) * trainingFrac));
	L7 = L7 (1 : floor(length(L7) * trainingFrac));
 
    if order == 'inorder'
		Inx1 = 1:1:length(L1);
		Inx2 = 1:1:length(L2);
		Inx3 = 1:1:length(L3);
		Inx4 = 1:1:length(L4);
		Inx5 = 1:1:length(L5);
		Inx6 = 1:1:length(L6);
		Inx7 = 1:1:length(L7);
	else
		Inx1 = randperm (length(L1));
		Inx2 = randperm (length(L2));
		Inx3 = randperm (length(L3));
		Inx4 = randperm (length(L4));
		Inx5 = randperm (length(L5));
		Inx6 = randperm (length(L6));
		Inx7 = randperm (length(L7));
	end
    
    L_SAMPL_DATA = labels (L1(Inx1));
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L2(Inx2)))];
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L3(Inx3)))];
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L4(Inx4)))];
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L5(Inx5)))];
	L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L6(Inx6)))];
	L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L7(Inx7)))];
    %L_SAMPL_DATA = L_SAMPL_DATA';
    
    SAMPL_DATA   = data (L1(Inx1), :);
    SAMPL_DATA   = [SAMPL_DATA; (data(L2(Inx2), :))];
    SAMPL_DATA   = [SAMPL_DATA; (data(L3(Inx3), :))];
    SAMPL_DATA   = [SAMPL_DATA; (data(L4(Inx4), :))];
    SAMPL_DATA   = [SAMPL_DATA; (data(L5(Inx5), :))];
	SAMPL_DATA   = [SAMPL_DATA; (data(L6(Inx6), :))];
	SAMPL_DATA   = [SAMPL_DATA; (data(L7(Inx7), :))];
end

function [downSampledData, downSampledLabels] = downSampling (data, labels, donwSampRate)
%
% DESCRIPTION   : apply a downsampling to get rid of redundancy in signals 
%
% INPUTS:
%   data        : input data
%   labels      : input labels
%   donwSampRate: the rate or stride of downsampling
% OUTPUTS:
%   downSampledData: downsampled data
%   downSampledLabels: downsampled labels
%    
	j = 1;
	for i = 1:donwSampRate:length(data)
		downSampledData (j,:) = data(i, :);
		downSampledLabels (j) = labels(i);
        j = j + 1;
    end
    
    downSampledLabels = downSampledLabels';
end
	
function randomHV = genRandomHV(D)
%
% DESCRIPTION   : generate a random vector with zero mean 
%
% INPUTS:
%   D           : Dimension of vectors
% OUTPUTS:
%   randomHV    : generated random vector

    if mod(D,2)
        disp ('Dimension is odd!!');
    else
        randomIndex = randperm (D);
        randomHV (randomIndex(1 : D/2)) = 1;
        randomHV (randomIndex(D/2+1 : D)) = -1;
    end
end


function [CiM, iM] = initItemMemories (D, MAXL)
%
% DESCRIPTION   : initialize the item Memory  
%
% INPUTS:
%   D           : Dimension of vectors
%   MAXL        : Maximum amplitude of EMG signal
% OUTPUTS:
%   iM          : item memory for IDs of channels
%   CiM         : continious item memory for value of a channel
 
    % MAXL = 21;
	CiM = containers.Map ('KeyType','double','ValueType','any');
	iM  = containers.Map ('KeyType','double','ValueType','any');
    rng('default');
    rng(1);
        
	%init 4 orthogonal vectors for the 4 channels
	iM(1) = genRandomHV (D);
	iM(2) = genRandomHV (D);
	iM(3) = genRandomHV (D);
	iM(4) = genRandomHV (D);

    initHV = genRandomHV (D);
	currentHV = initHV;
	randomIndex = randperm (D);
	
    for i = 0:1:MAXL
        CiM(i) = currentHV;
		%D / 2 / MAXL = 238
        SP = floor(D/2/MAXL);
		startInx = (i*SP) + 1;
		endInx = ((i+1)*SP) + 1;
		currentHV (randomIndex(startInx : endInx)) = currentHV (randomIndex(startInx: endInx)) * -1;
    end
end


function randomHV = lookupItemMemeory (itemMemory, rawKey, D, percision)
%
% DESCRIPTION   : recalls a vector from item Memory based on inputs
%
% INPUTS:
%   itemMemory  : item memory
%   rawKey      : the input key
%   D           : Dimension of vectors
%   percision   : percision used in quantization of input EMG signals
%
% OUTPUTS:
%   randomHV    : return the related vector

 
    key = int64 (rawKey * percision);
    if itemMemory.isKey (key) 
        randomHV = itemMemory (key);
    else
        fprintf ('CANNOT FIND THIS KEY: %d\n', key);       
    end
end


function Ngram = computeNgram (buffer, CiM, D, N, percision, iM)
    %init
    ch1HV = zeros (1,D);
	ch2HV = zeros (1,D);
	ch3HV = zeros (1,D);
	ch4HV = zeros (1,D);
	record = zeros (1,D);
	Ngram = zeros (1,D);
	
	ch1HV = lookupItemMemeory (CiM, buffer(1, 1), D, percision);
	ch2HV = lookupItemMemeory (CiM, buffer(1, 2), D, percision);
	ch3HV = lookupItemMemeory (CiM, buffer(1, 3), D, percision);
	ch4HV = lookupItemMemeory (CiM, buffer(1, 4), D, percision);
	ch1HV = ch1HV .* iM(1);
	ch2HV = ch2HV .* iM(2);
	ch3HV = ch3HV .* iM(3);
	ch4HV = ch4HV .* iM(4);
	Ngram = ch1HV + ch2HV + ch3HV + ch4HV;
	
	for i = 2:1:N
		ch1HV = lookupItemMemeory (CiM, buffer(i, 1), D, percision);
		ch2HV = lookupItemMemeory (CiM, buffer(i, 2), D, percision);
		ch3HV = lookupItemMemeory (CiM, buffer(i, 3), D, percision);
		ch4HV = lookupItemMemeory (CiM, buffer(i, 4), D, percision);
		ch1HV = ch1HV .* iM(1);
		ch2HV = ch2HV .* iM(2);
		ch3HV = ch3HV .* iM(3);
		ch4HV = ch4HV .* iM(4);
		record = ch1HV + ch2HV + ch3HV + ch4HV;
		Ngram = circshift (Ngram, [1,1]) .* record;
	end
end

function sumHV = computeSumHV (buffer, CiM, D, N, percision, iM)
	sumHV = zeros (1,D);
    for i = 1:1:length(buffer(:,1))-N+1
		newNgram = computeNgram (buffer(i:i+N-1,:), CiM, D, N, percision, iM);
		sumHV = sumHV + newNgram;
    end
end

function [numPat, AM] = hdctrain (labelTrainSet, trainSet, CiM, iM, D, N, percision, cuttingAngle) 
%
% DESCRIPTION   : train an associative memory based on input training data
%
% INPUTS:
%   labelTrainSet : training labels
%   trainSet    : EMG training data
%   CiM         : cont. item memory
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size
%   percision   : percision used in quantization of input EMG signals
%   cuttingAngle: threshold angle for not including a vector into SUM vector
%
% OUTPUTS:
%   AM          : Trained associative memory
%   numPat      : Number of stored patterns for each class of AM
%
 
	AM = containers.Map ('KeyType','double','ValueType','any');
	fprintf ('Total traning samples size = %d\n', length(labelTrainSet));
	
    for label = 1:1:max(labelTrainSet)
    	AM (label) = zeros (1,D);
	    numPat (label) = 0;
    end

    i = 1;
    while i < length(labelTrainSet)-N+1
       	if labelTrainSet(i) == labelTrainSet (i+N-1)			
			ngram = computeNgram (trainSet (i : i+N-1,:), CiM, D, N, percision, iM);
			angle = cosAngle(ngram, AM (labelTrainSet (i+N-1)));
			if angle < cuttingAngle | isnan(angle)
				AM (labelTrainSet (i+N-1)) = AM (labelTrainSet (i+N-1)) + ngram;
	            numPat (labelTrainSet (i+N-1)) = numPat (labelTrainSet (i+N-1)) + 1;
			end
            i = i + 1;
		else
            i = i+N-1;
        end
    end
        
    for label = 1:1:max(labelTrainSet)
		fprintf ('Class= %d \t mean= %.0f \t created \n', label, mean(AM(label)));
    end
end

function [accExcTrnz, accuracy] = hdcpredict (labelTestSet, testSet, AM, CiM, iM, D, N, percision)
%
% DESCRIPTION   : test accuracy based on input testing data
%
% INPUTS:
%   labelTestSet: testing labels
%   testSet     : EMG test data
%   AM          : Trained associative memory
%   CiM         : Cont. item memory
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size
%   percision   : percision used in quantization of input EMG signals
%
% OUTPUTS:
%   accuracy    : classification accuracy for all situations
%   accExcTrnz  : classification accuracy excluding the transitions between gestutes
%
	correct = 0;
    numTests = 0;
	tranzError = 0;
	
    %for i = 1:1:length(testSet)-N+1
	for i = 1:N:length(testSet)-N+1
		numTests = numTests + 1;
		actualLabel = mode(labelTestSet (i : i+N-1));
    
		sigHV = computeSumHV (testSet (i : i+N-1,:), CiM, D, N, percision, iM);
		maxAngle = -1;
        predicLabel = -1;
		for label = 1:1:max(labelTestSet)
			angle = cosAngle(AM (label), sigHV);
			if (angle > maxAngle)
				maxAngle = angle;
				predicLabel = label;
			end
        end
        
		if predicLabel == actualLabel
			correct = correct + 1;
        elseif labelTestSet (i) ~= labelTestSet(i+N-1)
			tranzError = tranzError + 1;
			%fprintf ('   !!! WRONG with ANGLE of %.2f !!! A[%d] --> P[%d]\n', maxAngle, actualLabel, predicLabel);
		end
    end

    accuracy = correct / numTests;
	accExcTrnz = (correct + tranzError) / numTests;
end


function [accuracy] = test_slicing (labels, data, AM, CiM, iM, D, N, percision, perOverLap)
%
% DESCRIPTION   : test accuracy based on input testing data but it automatically slices the test signal to different gestures
%
% INPUTS:
%   labels      : testing labels
%   data        : EMG test data
%   AM          : Trained associative memory
%   CiM         : Cont. item memory
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size
%   percision   : percision used in quantization of input EMG signals
%   perOverLap  : percentage of overlapping between adjacent windows
%
% OUTPUTS:
%   accuracy    : classification accuracy when using sliced windows
%


	correct = 0;
    numTests = 0;

    start = length(labels)-1;	
    for i = 1:1:length(labels)-1
         if (labels (i) == labels (i+1)) & (start > i)
            start = i;
        elseif (labels (i) ~= labels (i+1)) & (start <= i)
            stop = i;
            window = stop - start;
            window = max (window, N);
			addedMargin = int32(window * perOverLap);
		
            if start-addedMargin < 1 | stop+window+addedMargin > length(labels)-1
        		fprintf (' !!!! selected gesture is out of range %d:%d !!! \n', start-addedMargin, stop+addedMargin);
            else
				[maxAngle, predicLabel] = hdcpredict_window_max (data(start-addedMargin : start+window+addedMargin, :), AM, CiM, iM, D, N, percision);
                	
		        numTests = numTests + 1;
                if predicLabel == labels(start)
			        correct = correct + 1;
                else
        		    fprintf (':-( %d went to %d : for signal range of %d:%d \n', labels(start), predicLabel, start, stop);
                end
            end
            start = length(data)-1;
        end
    end
    accuracy = correct / numTests;
end


function [maxAngle, predicLabel] = hdcpredict_window_max (testSet, AM, CiM, iM, D, N, percision)
%
% DESCRIPTION   : predicts a label for an ngram that has the highest similarity within a fixed window size 
%
% INPUTS:
%   testSet     : The block of data in which this function searches to find the highest similarity
%   AM          : Trained associative memory
%   CiM         : Cont. item memory
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size
%   percision   : percision used in quantization of input EMG signals
%
% OUTPUTS:
%   maxAngle    : the maximum angle of ngram found in the window
%   predicLabel : predicted label for the ngram
%
    maxAngle = -1;
    predicLabel = -1;
    for i = 1:1:length(testSet(:,1))-N+1
		sigHV = computeSumHV (testSet(i : i+N-1,:), CiM, D, N, percision, iM);
	    for label = 1:1:5
			angle = cosAngle (AM(label), sigHV);
			if (angle > maxAngle)
				maxAngle = angle;
				predicLabel = label;
			end
        end
    end
end

function cosAngle = cosAngle (u, v)
    cosAngle = dot(u,v)/(norm(u)*norm(v));
end



function [meanAngles, stdAngles] = findbestN (labels, data, emgAM, iMch, chAM, D, percision, maxN)
    for N = 1:1:maxN
        angleSet = 0;
        start = length(labels)-1;
        for i = 1:1:length(labels)-1
             if (labels (i) == labels (i+1)) & (start > i)
                start = i;
            elseif (labels (i) ~= labels (i+1)) & (start <= i)
                stop = i;
                window = stop - start;
                window = max (window, N);
                %if stop-start < N
                %   fprintf ('TOO small test window gesture of %d for N-gram of %d \n', stop-start, N);
                %else
                [maxAngle, predicLabel] = hdcpredict_window_max (data(start:start+window, :), emgAM, iMch, chAM, D, N, percision);
                angleSet = [angleSet; maxAngle];
                %end
                start = length(data)-1;
                %break;
            end
        end
        angleSet (1) = [];
        meanAngles (N) = mean (angleSet);
        stdAngles  (N) = std (angleSet);
    end
    [m bestN] = max (meanAngles);
end


