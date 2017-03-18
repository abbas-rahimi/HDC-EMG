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

function message = binaryCode
  assignin('base','lookupItemMemeory', @lookupItemMemeory);
  assignin('base','genRandomHV', @genRandomHV);
  assignin('base','cosAngle', @cosAngle);
  assignin('base','computeNgram_4N_features', @computeNgram_4N_features);
  assignin('base','computeNgram_1N_feature_perm', @computeNgram_1N_feature_perm);
  assignin('base','hdctrain', @hdctrain); 
  assignin('base','initItemMemories', @initItemMemories);
  assignin('base','genTrainData', @genTrainData);
  assignin('base','downSampling', @downSampling);
  assignin('base','hdcpredict', @hdcpredict);
  
  assignin('base','test_slicing', @test_slicing);
  assignin('base','hdcpredict_window', @hdcpredict_window);
  message='Done importing functions to workspace';
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

        global codingMode;
	    switch codingMode
            case 'dense_bipolar'
                randomHV (randomIndex(D/2+1 : D)) = -1;
            case 'dense_binary'
                randomHV (randomIndex(D/2+1 : D)) = 0;
        end
    end
end

function [iM] = initItemMemories (D, MAXL, N)
%
% DESCRIPTION   : initialize the item Memory (here continous) 
%
% INPUTS:
%   D           : Dimension of vectors
%   MAXL        : Maximum amplitude of EMG signal
%   N           : size of n-gram, i.e., window size
% OUTPUTS:
%   iM          : item memory
 
	iM = containers.Map ('KeyType','char','ValueType','any');
    rng ('default');
    rng (1);
    
    % Here we form N*4 features, hence we require N*4 orthogonal initial vectors        
    for j = 1:1:N*4
	    currentHV = genRandomHV (D);
	    randomIndex = randperm (D);
        % Iterate over dicrete levels and 'continuously' generate vectors for related values	
        for i = 0:1:MAXL
            key = strcat(int2str(i),'x',int2str(j));
            iM(key) = currentHV;
		    %D / 2 / MAXL = 238
            SP = floor(D/2/MAXL);
            % start index and end index for flipping bits
		    startInx = (i*SP) + 1;
		    endInx = ((i+1)*SP) + 1;
            % flip these random bits
            
            global codingMode;
	        switch codingMode
                case 'dense_bipolar'
		            currentHV (randomIndex(startInx : endInx)) = currentHV (randomIndex(startInx : endInx)) * -1;
                case 'dense_binary'
		            currentHV (randomIndex(startInx : endInx)) = not (currentHV (randomIndex(startInx : endInx)));
            end
        end
    end
end


function randomHV = lookupItemMemory (itemMemory, channelValue, channelID, percision)
%
% DESCRIPTION   : recalls a vector from item Memory (here continous) based on inputs
%
% INPUTS:
%   itemMemory  : item memory
%   channelValue: the analog value of EMG channel
%   channelID   : ID of EMG channel from 1 to 4*N
%   percision   : percision used in quantization of input EMG signals
%
% OUTPUTS:
%   randomHV    : return the related vector

    % Discretization of EMG signal based on required percision
    quantized = int64 (channelValue * percision);
    % generates a key to find related vector for quantized value in channel with this ID
    key = strcat (int2str(quantized), 'x', int2str(channelID));
    if itemMemory.isKey (key) 
        randomHV = itemMemory (key);
        %fprintf ('READING KEY: %s\n', key);
    else
        fprintf ('CANNOT FIND THIS KEY: %s\n', key);       
    end
end

function ngram = computeNgram_4N_features (block, iM, D, N, percision)
%
% DESCRIPTION   : compute an ngram vector for a block of input data; a block has 4*N sampels data
%               This encoding uses 4*N separate features and hence 4*N item memories
%
% INPUTS:
%   block       : input data
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size
%   percision   : percision used in quantization of input EMG signals
%
% OUTPUTS:
%   ngram       : the ngram vector

    global codingMode;
	switch codingMode
        case 'dense_bipolar'
            ngram = zeros (1,D);
            % Go over various time stamps to capture signal history (n-gram)
            for t = 1:1:N
            % Go over the 4 channels of EMG
                for c = 1:1:4
			        ngram = ngram + lookupItemMemory (iM, block(t,c), 4*(t-1)+c, percision);
                end
            end

        case 'dense_binary'
            ngram = zeros (1,D);
            % Go over various time stamps to capture signal history (n-gram)
            for t = 1:1:N
            % Go over the 4 channels of EMG
                for c = 1:1:4
                    retrieveVec = lookupItemMemory (iM, block(t,c), 4*(t-1)+c, percision);
			        ngram = [ngram; retrieveVec];
                end
            end
            ngram (1,:) = [];
            ngram = mode(ngram);
    end
end

 
function ngram = computeNgram_1N_feature_perm (block, iM, D, N, percision)
%
% DESCRIPTION   : compute an ngram vector for a block of input data; a block has 4*N sampels data
%               This encoding uses 4 separate features and hence 4 item memories and then applies permutation
% INPUTS:
%   block       : input data
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size
%   percision   : percision used in quantization of input EMG signals
%
% OUTPUTS:
%   ngram       : the ngram vector
    
    global codingMode;
	switch codingMode
        case 'dense_bipolar'
        	ngram = zeros (1,D);
            % Go over various time stamps to capture signal history (n-gram)
            for t = 1:1:N
                % Compute a sum vector for each time stamp of signal, i.e., a vertical slicing of EMG data (N=1)
	            sumv = zeros (1,D);
                for c = 1:1:4
			        sumv = sumv + lookupItemMemory (iM, block(t,c), c, percision);
                end
	            % Now, permure it to compute n-gram = sumv(1) + p(sumv(2)) + pp(sumv(3) + ... 
                ngram = ngram + circshift (sumv, [1, t-1]);
            end

        case 'dense_binary'
            ngram = zeros (1,D);
            % Go over various time stamps to capture signal history (n-gram)
            for t = 1:1:N
                % Compute a sum vector for each time stamp of signal, i.e., a vertical slicing of EMG data (N=1)
	            sumv = zeros (1,D);
                for c = 1:1:4
                    retrieveVec = lookupItemMemory (iM, block(t,c), c, percision);
			        sumv = [sumv; retrieveVec];
                end
                sumv (1,:) = [];
                sumv = mode (sumv);
	            % Now, permure it to compute n-gram = sumv(1) + p(sumv(2)) + pp(sumv(3) + ... 
                rotatedVec = circshift (sumv, [1, t-1]);
                ngram = [ngram; rotatedVec];
            end
            ngram(1,:) = [];
            ngram = mode(ngram);
    end
end

             
function [numPat, AM] = hdctrain (labels, data, iM, D, N, percision, cuttingAngle) 
%
% DESCRIPTION   : train an associative memory based on input training data
%
% INPUTS:
%   labels      : training labels
%   data        : EMG training data
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
    global trainingMode;
	switch trainingMode
        case 'integer_vect'
     		% initialize an empty AM
			AM = containers.Map ('KeyType','double','ValueType','any');
			fprintf ('Total traning samples size = %d\n', length(labels));
			for label = 1:1:max(labels)
				AM (label) = zeros (1,D);
				numPat (label) = 0;
			end

			% walk through the input data 
			i = 1;
			while i < length(labels)-N+1
				% if labels for the entire of n-gram (or window) are the same; it excludes training samples that are in transitions between 2 neighbor gestures 
				if labels(i) == labels(i+N-1)		
					% compute n-gram; the encoderMode determines the type of encoding algorithm
					global encoderMode;
					switch encoderMode
						case '4N_features'
							ngram = computeNgram_4N_features (data(i : i+N-1,:), iM, D, N, percision);
						case 'N_feature_perm'
							ngram = computeNgram_1N_feature_perm (data(i : i+N-1,:), iM, D, N, percision);
					end

					% check whether the produced n-gram is already in AM
					angle = cosAngle(ngram, AM (labels(i+N-1)));
					if angle < cuttingAngle | isnan(angle)
						AM (labels(i+N-1)) = AM (labels(i+N-1)) + ngram;
						numPat (labels(i+N-1)) = numPat (labels(i+N-1)) + 1;
					end
					% walk through data with stride 1
					i = i + 1;
				else
					% jump to the next window that has a 'stable' gesture where all labels are equal
					i = i + N - 1;
				end
			end
		
		case 'binary_vect'
     		% initialize an empty AM
			AM = containers.Map ('KeyType','double','ValueType','any');
			fprintf ('Total traning samples size = %d\n', length(labels));
			for label = 1:1:max(labels)
				%AM (label) = zeros (1,D);
				numPat (label) = 0;
				trainVecList = zeros (1,D);
				% walk through the input data 
				i = 1;
				while i < length(labels)-N+1
					% if labels for the entire of n-gram (or window) are the same; it excludes training samples that are in transitions between 2 neighbor gestures 
					if (labels(i) == labels(i+N-1)) && (labels(i) == label)		
						% compute n-gram; the encoderMode determines the type of encoding algorithm
						global encoderMode;
						switch encoderMode
							case '4N_features'
								ngram = computeNgram_4N_features (data(i : i+N-1,:), iM, D, N, percision);
							case 'N_feature_perm'
								ngram = computeNgram_1N_feature_perm (data(i : i+N-1,:), iM, D, N, percision);
						end

						% check whether the produced n-gram is already in AM
						%angle = cosAngle(ngram, mode (trainVecList));
						%if angle < cuttingAngle | isnan(angle)
							trainVecList = [trainVecList; ngram];
							numPat (labels(i+N-1)) = numPat (labels(i+N-1)) + 1;
						%end
						% walk through data with stride 1
						i = i + 1;
					else
						% jump to the next window that has a 'stable' gesture where all labels are equal
						i = i + N - 1;
					end
				end
				trainVecList(1,:) = [];
				AM (label) = mode (trainVecList);
			end
    end	
			
    for label = 1:1:max(labels)
		fprintf ('Class= %d \t mean= %0.3f \t n_pat=%d \t created \n', label, mean(AM(label)), numPat(label));
    end
end
              

function [accExcTrnz, accuracy] = hdcpredict (labelTestSet, testSet, AM, iM, D, N, percision)
%
% DESCRIPTION   : test accuracy based on input testing data
%
% INPUTS:
%   labelTestSet: testing labels
%   testSet     : EMG test data
%   AM          : Trained associative memory
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size
%   percision   : percision used in quantization of input EMG signals
%
% OUTPUTS:
%   accuracy    : classification accuracy for all situations
%   accExcTrnz  : classification accuracy excluding the transitions between gestutes
%
	correct = 0;        % count the number of correct classifications
    numTests = 0;       % count the total number of tests
	tranzError = 0;     % count the number of tests that went wrong but were between gesture transitions 
	
    % Go over the test data with stride of N
	for i = 1:N:length(testSet)-N+1
		numTests = numTests + 1;
		% set actual label as the majority of labels in the window!
        actualLabel = mode(labelTestSet (i : i+N-1));
        
        % compute ngram vector for the testing window; the encoderMode determines the type of encoding algorithm   
        global encoderMode;
		switch encoderMode
            case '4N_features'
		        sigHV = computeNgram_4N_features (testSet (i : i+N-1,:), iM, D, N, percision);
            case 'N_feature_perm'
		        sigHV = computeNgram_1N_feature_perm (testSet (i : i+N-1,:), iM, D, N, percision);
        end

        % find the label of testing window
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
		end
    end

    accuracy = correct / numTests;
	accExcTrnz = (correct + tranzError) / numTests;
end


function [accuracy] = test_slicing (labels, data, AM, iM, D, N, percision)
%
% DESCRIPTION   : test accuracy based on input testing data but it automatically slices the test signal to different gestures
%
% INPUTS:
%   labels      : testing labels
%   data        : EMG test data
%   AM          : Trained associative memory
%   iM          : item memory
%   D           : Dimension of vectors
%   N           : size of n-gram, i.e., window size
%   percision   : percision used in quantization of input EMG signals
%
% OUTPUTS:
%   accuracy    : classification accuracy when using sliced windows
%

	correct = 0;
    numTests = 0;
    start = length(labels)-1;	
    
    % Go over the test data with stride of 1 and finds a window during which all labels are equal (hence, set start and stop indices) 
    for i = 1:1:length(labels)-1
         if (labels (i) == labels (i+1)) & (start > i)
            start = i;
         elseif (labels (i) ~= labels (i+1)) & (start <= i)
            stop = i;
            window = stop - start;
            window = max (window, N);
		
            if start < 1 | stop+window > length(labels)-1
        		fprintf (' !!!! selected gesture is out of range %d:%d !!! \n', start, stop);
            else
                % Find the label for this window (that might have multiple ngrams) by picking up the ngram that has highest similarity
				[maxAngle, predicLabel] = hdcpredict_window_max (data(start : start+window, :), AM, iM, D, N, percision);
                	
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

function [maxAngle, predicLabel] = hdcpredict_window_max (testSet, AM, iM, D, N, percision)
%
% DESCRIPTION   : predicts a label for an ngram that has the highest similarity within a fixed window size 
%
% INPUTS:
%   testSet     : The block of data in which this function searches to find the highest similarity
%   AM          : Trained associative memory
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

    % Walk through the block of data (that might have multiple ngrams) with stride 1
    for i = 1:1:length(testSet(:,1))-N+1
        % compute ngram vectors that are available in  the testing window; the encoderMode determines the type of encoding algorithm   
        global encoderMode;
		switch encoderMode
            case '4N_features'
		        sigHV = computeNgram_4N_features (testSet(i : i+N-1,:), iM, D, N, percision);
            case 'N_feature_perm'
        		sigHV = computeNgram_1N_feature_perm (testSet(i : i+N-1,:), iM, D, N, percision);
        end
        % Pick up the label for ngram that has the maximum simularity with stored patterns in AM
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


