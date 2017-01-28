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

%load EMG dataset and set HD parameters
clear;
load('dataset.mat');
ICRC;
learningFrac = 0.25;
downSampRate = 1;
D = 10000;
percision = 1;
[iMch, chAM] = initItemMemories (D, 21);
cuttingAngle = 0.9;

[TS_COMPLETE_1, L_TS_COMPLETE_1] = downSampling (COMPLETE_1, LABEL_1, downSampRate);
[TS_COMPLETE_2, L_TS_COMPLETE_2] = downSampling (COMPLETE_2, LABEL_2, downSampRate);
[TS_COMPLETE_3, L_TS_COMPLETE_3] = downSampling (COMPLETE_3, LABEL_3, downSampRate);
[TS_COMPLETE_4, L_TS_COMPLETE_4] = downSampling (COMPLETE_4, LABEL_4, downSampRate);
[TS_COMPLETE_5, L_TS_COMPLETE_5] = downSampling (COMPLETE_5, LABEL_5, downSampRate);

[L_SAMPL_DATA_1, SAMPL_DATA_1] = genTrainData (TS_COMPLETE_1, L_TS_COMPLETE_1, learningFrac, '-------');
[L_SAMPL_DATA_2, SAMPL_DATA_2] = genTrainData (TS_COMPLETE_2, L_TS_COMPLETE_2, learningFrac, '-------');
[L_SAMPL_DATA_3, SAMPL_DATA_3] = genTrainData (TS_COMPLETE_3, L_TS_COMPLETE_3, learningFrac, '-------');
[L_SAMPL_DATA_4, SAMPL_DATA_4] = genTrainData (TS_COMPLETE_4, L_TS_COMPLETE_4, learningFrac, '-------');
[L_SAMPL_DATA_5, SAMPL_DATA_5] = genTrainData (TS_COMPLETE_5, L_TS_COMPLETE_5, learningFrac, '-------');

%=====================================================================================================================================
%SAMPLE-BY-SAMPLE
N = 1;
[numpat_1, hdc_model_1] = hdctrain (L_SAMPL_DATA_1, SAMPL_DATA_1, iMch, chAM, D, N, percision, cuttingAngle);
[accExcTrnz, acc_hdc_1] = hdcpredict (L_TS_COMPLETE_1, TS_COMPLETE_1, hdc_model_1, iMch, chAM, D, N, percision)

[numpat_2, hdc_model_2] = hdctrain (L_SAMPL_DATA_2, SAMPL_DATA_2, iMch, chAM, D, N, percision, cuttingAngle);
[accExcTrnz, acc_hdc_2] = hdcpredict (L_TS_COMPLETE_2, TS_COMPLETE_2, hdc_model_2, iMch, chAM, D, N, percision)

[numpat_3, hdc_model_3] = hdctrain (L_SAMPL_DATA_3, SAMPL_DATA_3, iMch, chAM, D, N, percision, cuttingAngle);
[accExcTrnz, acc_hdc_3] = hdcpredict (L_TS_COMPLETE_3, TS_COMPLETE_3, hdc_model_3, iMch, chAM, D, N, percision)

[numpat_4, hdc_model_4] = hdctrain (L_SAMPL_DATA_4, SAMPL_DATA_4, iMch, chAM, D, N, percision, cuttingAngle);
[accExcTrnz, acc_hdc_4] = hdcpredict (L_TS_COMPLETE_4, TS_COMPLETE_4, hdc_model_4, iMch, chAM, D, N, percision)

[numpat_5, hdc_model_5] = hdctrain (L_SAMPL_DATA_5, SAMPL_DATA_5, iMch, chAM, D, N, percision, cuttingAngle);
[accExcTrnz, acc_hdc_5] = hdcpredict (L_TS_COMPLETE_5, TS_COMPLETE_5, hdc_model_5, iMch, chAM, D, N, percision)

svm_model_1 = svmtrain (L_SAMPL_DATA_1, SAMPL_DATA_1, ' -c 500 -h 0');
[a, acc_svm_1, b] = svmpredict (L_TS_COMPLETE_1, TS_COMPLETE_1, svm_model_1); 

svm_model_2 = svmtrain (L_SAMPL_DATA_2, SAMPL_DATA_2, ' -c 500 -h 0');
[a, acc_svm_2, b] = svmpredict (L_TS_COMPLETE_2, TS_COMPLETE_2, svm_model_2); 

svm_model_3 = svmtrain (L_SAMPL_DATA_3, SAMPL_DATA_3, ' -c 500 -h 0');
[a, acc_svm_3, b] = svmpredict (L_TS_COMPLETE_3, TS_COMPLETE_3, svm_model_3); 

svm_model_4 = svmtrain (L_SAMPL_DATA_4, SAMPL_DATA_4, ' -c 500 -h 0');
[a, acc_svm_4, b] = svmpredict (L_TS_COMPLETE_4, TS_COMPLETE_4, svm_model_4); 

svm_model_5 = svmtrain (L_SAMPL_DATA_5, SAMPL_DATA_5, ' -c 500 -h 0');
[a, acc_svm_5, b] = svmpredict (L_TS_COMPLETE_5, TS_COMPLETE_5, svm_model_5); 

Y_svm = [acc_svm_1(1) acc_svm_2(1) acc_svm_3(1) acc_svm_4(1) acc_svm_5(1)];
Y_svm = [Y_svm mean(Y_svm)];
Y_hdc = [acc_hdc_1 acc_hdc_2 acc_hdc_3 acc_hdc_4 acc_hdc_5];
Y_hdc = [Y_hdc mean(Y_hdc)];
% It produces Figure 5 of the paper
errorbar_groups ([Y_svm; 100*Y_hdc], [Y_svm-Y_svm; Y_hdc-Y_hdc]);
ax = gca;
ax.YLabel.String = 'Accuracy (%)';
ax.YLim = [50 101];
legend('SVM', 'HDC', 'location', 'best'); 
ax.XTickLabel ={'S1', 'S2', 'S3', 'S4', 'S5', 'Mean'};
ax.XLabel.String = 'Different subjects';
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
%================================================================================================
%Measuring gestures statistics
%[meanWin_1 , stdWin_1] = measure_windowSize (L_TS_COMPLETE_1, TS_COMPLETE_1);

%================================================================================================
%EXPLORING N-grams
downSampRate = 250;
overlap = 0;
[TS_COMPLETE_1, L_TS_COMPLETE_1] = downSampling (COMPLETE_1, LABEL_1, downSampRate);
[TS_COMPLETE_2, L_TS_COMPLETE_2] = downSampling (COMPLETE_2, LABEL_2, downSampRate);
[TS_COMPLETE_3, L_TS_COMPLETE_3] = downSampling (COMPLETE_3, LABEL_3, downSampRate);
[TS_COMPLETE_4, L_TS_COMPLETE_4] = downSampling (COMPLETE_4, LABEL_4, downSampRate);
[L_SAMPL_DATA_1, SAMPL_DATA_1] = genTrainData (TS_COMPLETE_1, L_TS_COMPLETE_1, learningFrac, '-------');
[L_SAMPL_DATA_2, SAMPL_DATA_2] = genTrainData (TS_COMPLETE_2, L_TS_COMPLETE_2, learningFrac, '-------');
[L_SAMPL_DATA_3, SAMPL_DATA_3] = genTrainData (TS_COMPLETE_3, L_TS_COMPLETE_3, learningFrac, '-------');
[L_SAMPL_DATA_4, SAMPL_DATA_4] = genTrainData (TS_COMPLETE_4, L_TS_COMPLETE_4, learningFrac, '-------');

downSampRate = 50;
[TS_COMPLETE_5, L_TS_COMPLETE_5] = downSampling (COMPLETE_5, LABEL_5, downSampRate);
[L_SAMPL_DATA_5, SAMPL_DATA_5] = genTrainData (TS_COMPLETE_5, L_TS_COMPLETE_5, learningFrac, '-------');

for N = 1:1:7
	[numpat, hdc_model_1] = hdctrain (L_SAMPL_DATA_1, SAMPL_DATA_1, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_1, TS_COMPLETE_1, hdc_model_1, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_n_1 (N) = acch;
	numpat_n_1 (N,:) = [numpat sum(numpat)];
	
	[numpat, hdc_model_2] = hdctrain (L_SAMPL_DATA_2, SAMPL_DATA_2, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_2, TS_COMPLETE_2, hdc_model_2, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_n_2 (N) = acch;
	numpat_n_2 (N,:) = [numpat sum(numpat)];

	[numpat, hdc_model_3] = hdctrain (L_SAMPL_DATA_3, SAMPL_DATA_3, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_3, TS_COMPLETE_3, hdc_model_3, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_n_3 (N) = acch;
	numpat_n_3 (N,:) = [numpat sum(numpat)];
	
	[numpat, hdc_model_4] = hdctrain (L_SAMPL_DATA_4, SAMPL_DATA_4, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_4, TS_COMPLETE_4, hdc_model_4, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_n_4 (N) = acch;
	numpat_n_4 (N,:) = [numpat sum(numpat)];
	
	[numpat, hdc_model_5] = hdctrain (L_SAMPL_DATA_5, SAMPL_DATA_5, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_5, TS_COMPLETE_5, hdc_model_5, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_n_5 (N) = acch;
	numpat_n_5 (N,:) = [numpat sum(numpat)];
end
 
for N = 1:1:7
	clear TR_EXTENDED_1 L_TR_EXTENDED_1 L_TR_ROTATED_1;
	TR_EXTENDED_1 = SAMPL_DATA_1;
	L_TR_ROTATED_1(:,1) = L_SAMPL_DATA_1;
	for j = 2:1:N
		TR_ROTATED_1 = circshift(SAMPL_DATA_1, -1*j+1, 1);
		L_TR_ROTATED_1(:,j) = circshift(L_SAMPL_DATA_1, -1*j+1, 1);
		TR_EXTENDED_1 = [TR_EXTENDED_1 TR_ROTATED_1];
	end
	L_TR_EXTENDED_1 = mode (L_TR_ROTATED_1, 2);
	svm_model_1 = svmtrain (L_TR_EXTENDED_1, TR_EXTENDED_1, ' -c 500 -h 0');
	num_sv_n_1 (N) = svm_model_1.totalSV;
	
	clear TS_EXTENDED_1 L_TS_EXTENDED_1 L_TS_ROTATED_1;
	TS_EXTENDED_1 = TS_COMPLETE_1;
	L_TS_ROTATED_1(:,1) = L_TS_COMPLETE_1;
	for j = 2:1:N
		TS_ROTATED_1 = circshift(TS_COMPLETE_1, -1*j+1, 1);
		L_TS_ROTATED_1(:,j) = circshift(L_TS_COMPLETE_1, -1*j+1, 1);
		TS_EXTENDED_1 = [TS_EXTENDED_1 TS_ROTATED_1];
	end
	L_TS_EXTENDED_1 = mode (L_TS_ROTATED_1, 2);
	[v1, accs, v2] = svmpredict (L_TS_EXTENDED_1, TS_EXTENDED_1, svm_model_1);
	acc_svm_n_1 (N) = accs(1)/100;
	
	
	clear TR_EXTENDED_2 L_TR_EXTENDED_2 L_TR_ROTATED_2;
	TR_EXTENDED_2 = SAMPL_DATA_2;
	L_TR_ROTATED_2(:,1) = L_SAMPL_DATA_2;
	for j = 2:1:N
		TR_ROTATED_2 = circshift(SAMPL_DATA_2, -1*j+1, 1);
		L_TR_ROTATED_2(:,j) = circshift(L_SAMPL_DATA_2, -1*j+1, 1);
		TR_EXTENDED_2 = [TR_EXTENDED_2 TR_ROTATED_2];
	end
	L_TR_EXTENDED_2 = mode (L_TR_ROTATED_2, 2);
	svm_model_2 = svmtrain (L_TR_EXTENDED_2, TR_EXTENDED_2, ' -c 500 -h 0');
	num_sv_n_2 (N) = svm_model_2.totalSV;
	
	clear TS_EXTENDED_2 L_TS_EXTENDED_2 L_TS_ROTATED_2;
	TS_EXTENDED_2 = TS_COMPLETE_2;
	L_TS_ROTATED_2(:,1) = L_TS_COMPLETE_2;
	for j = 2:1:N
		TS_ROTATED_2 = circshift(TS_COMPLETE_2, -1*j+1, 1);
		L_TS_ROTATED_2(:,j) = circshift(L_TS_COMPLETE_2, -1*j+1, 1);
		TS_EXTENDED_2 = [TS_EXTENDED_2 TS_ROTATED_2];
	end
	L_TS_EXTENDED_2 = mode (L_TS_ROTATED_2, 2);
	[v1, accs, v2] = svmpredict (L_TS_EXTENDED_2, TS_EXTENDED_2, svm_model_2);
	acc_svm_n_2 (N) = accs(1)/100;
	
	clear TR_EXTENDED_3 L_TR_EXTENDED_3 L_TR_ROTATED_3;
	TR_EXTENDED_3 = SAMPL_DATA_3;
	L_TR_ROTATED_3(:,1) = L_SAMPL_DATA_3;
	for j = 2:1:N
		TR_ROTATED_3 = circshift(SAMPL_DATA_3, -1*j+1, 1);
		L_TR_ROTATED_3(:,j) = circshift(L_SAMPL_DATA_3, -1*j+1, 1);
		TR_EXTENDED_3 = [TR_EXTENDED_3 TR_ROTATED_3];
	end
	L_TR_EXTENDED_3 = mode (L_TR_ROTATED_3, 2);
	svm_model_3 = svmtrain (L_TR_EXTENDED_3, TR_EXTENDED_3, ' -c 500 -h 0');
	num_sv_n_3 (N) = svm_model_3.totalSV;
	
	clear TS_EXTENDED_3 L_TS_EXTENDED_3 L_TS_ROTATED_3;
	TS_EXTENDED_3 = TS_COMPLETE_3;
	L_TS_ROTATED_3(:,1) = L_TS_COMPLETE_3;
	for j = 2:1:N
		TS_ROTATED_3 = circshift(TS_COMPLETE_3, -1*j+1, 1);
		L_TS_ROTATED_3(:,j) = circshift(L_TS_COMPLETE_3, -1*j+1, 1);
		TS_EXTENDED_3 = [TS_EXTENDED_3 TS_ROTATED_3];
	end
	L_TS_EXTENDED_3 = mode (L_TS_ROTATED_3, 2);
	[v1, accs, v2] = svmpredict (L_TS_EXTENDED_3, TS_EXTENDED_3, svm_model_3);
	acc_svm_n_3 (N) = accs(1)/100;
	
	
	clear TR_EXTENDED_4 L_TR_EXTENDED_4 L_TR_ROTATED_4;
	TR_EXTENDED_4 = SAMPL_DATA_4;
	L_TR_ROTATED_4(:,1) = L_SAMPL_DATA_4;
	for j = 2:1:N
		TR_ROTATED_4 = circshift(SAMPL_DATA_4, -1*j+1, 1);
		L_TR_ROTATED_4(:,j) = circshift(L_SAMPL_DATA_4, -1*j+1, 1);
		TR_EXTENDED_4 = [TR_EXTENDED_4 TR_ROTATED_4];
	end
	L_TR_EXTENDED_4 = mode (L_TR_ROTATED_4, 2);
	svm_model_4 = svmtrain (L_TR_EXTENDED_4, TR_EXTENDED_4, ' -c 500 -h 0');
	num_sv_n_4 (N) = svm_model_4.totalSV;
	
	clear TS_EXTENDED_4 L_TS_EXTENDED_4 L_TS_ROTATED_4;
	TS_EXTENDED_4 = TS_COMPLETE_4;
	L_TS_ROTATED_4(:,1) = L_TS_COMPLETE_4;
	for j = 2:1:N
		TS_ROTATED_4 = circshift(TS_COMPLETE_4, -1*j+1, 1);
		L_TS_ROTATED_4(:,j) = circshift(L_TS_COMPLETE_4, -1*j+1, 1);
		TS_EXTENDED_4 = [TS_EXTENDED_4 TS_ROTATED_4];
	end
	L_TS_EXTENDED_4 = mode (L_TS_ROTATED_4, 2);
	[v1, accs, v2] = svmpredict (L_TS_EXTENDED_4, TS_EXTENDED_4, svm_model_4);
	acc_svm_n_4 (N) = accs(1)/100;
	
	clear TR_EXTENDED_5 L_TR_EXTENDED_5 L_TR_ROTATED_5;
	TR_EXTENDED_5 = SAMPL_DATA_5;
	L_TR_ROTATED_5(:,1) = L_SAMPL_DATA_5;
	for j = 2:1:N
		TR_ROTATED_5 = circshift(SAMPL_DATA_5, -1*j+1, 1);
		L_TR_ROTATED_5(:,j) = circshift(L_SAMPL_DATA_5, -1*j+1, 1);
		TR_EXTENDED_5 = [TR_EXTENDED_5 TR_ROTATED_5];
	end
	L_TR_EXTENDED_5 = mode (L_TR_ROTATED_5, 2);
	svm_model_5 = svmtrain (L_TR_EXTENDED_5, TR_EXTENDED_5, ' -c 500 -h 0');
	num_sv_n_5 (N) = svm_model_5.totalSV;
	
	clear TS_EXTENDED_5 L_TS_EXTENDED_5 L_TS_ROTATED_5;
	TS_EXTENDED_5 = TS_COMPLETE_5;
	L_TS_ROTATED_5(:,1) = L_TS_COMPLETE_5;
	for j = 2:1:N
		TS_ROTATED_5 = circshift(TS_COMPLETE_5, -1*j+1, 1);
		L_TS_ROTATED_5(:,j) = circshift(L_TS_COMPLETE_5, -1*j+1, 1);
		TS_EXTENDED_5 = [TS_EXTENDED_5 TS_ROTATED_5];
	end
	L_TS_EXTENDED_5 = mode (L_TS_ROTATED_5, 2);
	[v1, accs, v2] = svmpredict (L_TS_EXTENDED_5, TS_EXTENDED_5, svm_model_5);
	acc_svm_n_5 (N) = accs(1)/100;		
end	

Y_svm = acc_svm_n_1;
Y_hdc = acc_hdc_n_1;

% These produce the set of plots for 5 subjects in Figure 7
errorbar_groups (100*[acc_svm_n_1; acc_hdc_n_1], [Y_svm-Y_svm; Y_hdc-Y_hdc]);
ax = gca;
ax.YLabel.String = 'Accuracy (%)';
ax.YLim = [50 101];
ax.XLabel.String = 'N-grams';
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
legend('SVM', 'HDC', 'location', 'southeast'); 
title('S1');

errorbar_groups (100*[acc_svm_n_2; acc_hdc_n_2], [Y_svm-Y_svm; Y_hdc-Y_hdc]);
ax = gca;
ax.YLabel.String = 'Accuracy (%)';
ax.YLim = [50 101];
ax.XLabel.String = 'N-grams';
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
legend('SVM', 'HDC', 'location', 'southeast'); 
title('S2');

errorbar_groups (100*[acc_svm_n_3; acc_hdc_n_3], [Y_svm-Y_svm; Y_hdc-Y_hdc]);
ax = gca;
ax.YLabel.String = 'Accuracy (%)';
ax.YLim = [50 101];
ax.XLabel.String = 'N-grams';
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
legend('SVM', 'HDC', 'location', 'southeast'); 
title('S3');

errorbar_groups (100*[acc_svm_n_4; acc_hdc_n_4], [Y_svm-Y_svm; Y_hdc-Y_hdc]);
ax = gca;
ax.YLabel.String = 'Accuracy (%)';
ax.YLim = [50 101];
ax.XLabel.String = 'N-grams';
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
legend('SVM', 'HDC', 'location', 'southeast'); 
title('S4');

errorbar_groups (100*[acc_svm_n_5; acc_hdc_n_5], [Y_svm-Y_svm; Y_hdc-Y_hdc]);
ax = gca;
ax.YLabel.String = 'Accuracy (%)';
ax.YLim = [50 101];
ax.XLabel.String = 'N-grams';
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
legend('SVM', 'HDC', 'location', 'southeast'); 
title('S5');

%=======================================================================================================================
%AUTOMATIC ADJUSTMENT OF WINDOW
N = 4;
[numpat, hdc_model_1] = hdctrain (L_SAMPL_DATA_1, SAMPL_DATA_1, iMch, chAM, D, N, percision, cuttingAngle);

N = 4;
[numpat, hdc_model_2] = hdctrain (L_SAMPL_DATA_2, SAMPL_DATA_2, iMch, chAM, D, N, percision, cuttingAngle);

N = 3;
[numpat, hdc_model_3] = hdctrain (L_SAMPL_DATA_3, SAMPL_DATA_3, iMch, chAM, D, N, percision, cuttingAngle);

N = 5;
[numpat, hdc_model_4] = hdctrain (L_SAMPL_DATA_4, SAMPL_DATA_4, iMch, chAM, D, N, percision, cuttingAngle);

N = 4;
[numpat, hdc_model_5] = hdctrain (L_SAMPL_DATA_5, SAMPL_DATA_5, iMch, chAM, D, N, percision, cuttingAngle);

maxN = 10;
[meanAngles_1, stdAngles_1] = findbestN (L_TS_COMPLETE_1, TS_COMPLETE_1, hdc_model_1, iMch, chAM, D, percision, maxN);
[meanAngles_2, stdAngles_2] = findbestN (L_TS_COMPLETE_2, TS_COMPLETE_2, hdc_model_2, iMch, chAM, D, percision, maxN);
[meanAngles_3, stdAngles_3] = findbestN (L_TS_COMPLETE_3, TS_COMPLETE_3, hdc_model_3, iMch, chAM, D, percision, maxN);
[meanAngles_4, stdAngles_4] = findbestN (L_TS_COMPLETE_4, TS_COMPLETE_4, hdc_model_4, iMch, chAM, D, percision, maxN);
[meanAngles_5, stdAngles_5] = findbestN (L_TS_COMPLETE_5, TS_COMPLETE_5, hdc_model_5, iMch, chAM, D, percision, maxN);

% These produce a set of plots for 5 subjects in Figure 8
errorbar_groups (meanAngles_1, stdAngles_1);
ax = gca;
ax.YLabel.String = 'Cosine similarity';
ax.YLim = [-0.2 1];
ax.XLabel.String = 'Tested N-grams';
ax.XTick = 1:1:10;
ax.XLim = [0.5 10.5];
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
title('S1: Trained with tetragrams (N=4)');

errorbar_groups (meanAngles_2, stdAngles_2);
ax = gca;
ax.YLabel.String = 'Cosine similarity';
ax.YLim = [-0.2 1];
ax.XLabel.String = 'Tested N-grams';
ax.XTick = 1:1:10;
ax.XLim = [0.5 10.5];
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
title('S2: Trained with tetragrams (N=4)');

errorbar_groups (meanAngles_3, stdAngles_3);
ax = gca;
ax.YLabel.String = 'Cosine similarity';
ax.YLim = [-0.2 1];
ax.XLabel.String = 'Tested N-grams';
ax.XTick = 1:1:10;
ax.XLim = [0.5 10.5];
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
title('S3: Trained with trigrams (N=3)');

errorbar_groups (meanAngles_4, stdAngles_4);
ax = gca;
ax.YLabel.String = 'Cosine similarity';
ax.YLim = [-0.2 1];
ax.XLabel.String = 'Tested N-grams';
ax.XTick = 1:1:10;
ax.XLim = [0.5 10.5];
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
title('S4: Trained with pentagrams (N=5)');

errorbar_groups (meanAngles_5, stdAngles_5);
ax = gca;
ax.YLabel.String = 'Cosine similarity';
ax.YLim = [-0.2 1];
ax.XLabel.String = 'Tested N-grams';
ax.XTick = 1:1:10;
ax.XLim = [0.5 10.5];
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
title('S5: Trained with tetragrams (N=4)');

%=====================================================================================================================
%EXPANDING THE SIZE OF TESTING WINDOW
for overlap = 1:1:100
    N = 4;
	[numpat, hdc_model_1] = hdctrain (L_SAMPL_DATA_1, SAMPL_DATA_1, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_1, TS_COMPLETE_1, hdc_model_1, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_ov_1 (overlap) = acch;
	numpat_ov_1 (overlap,:) = [numpat sum(numpat)];
	
    N = 4;
	[numpat, hdc_model_2] = hdctrain (L_SAMPL_DATA_2, SAMPL_DATA_2, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_2, TS_COMPLETE_2, hdc_model_2, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_ov_2 (overlap) = acch;
	numpat_ov_2 (overlap,:) = [numpat sum(numpat)];

    N = 3;
	[numpat, hdc_model_3] = hdctrain (L_SAMPL_DATA_3, SAMPL_DATA_3, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_3, TS_COMPLETE_3, hdc_model_3, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_ov_3 (overlap) = acch;
	numpat_ov_3 (overlap,:) = [numpat sum(numpat)];
	
    N = 5;
	[numpat, hdc_model_4] = hdctrain (L_SAMPL_DATA_4, SAMPL_DATA_4, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_4, TS_COMPLETE_4, hdc_model_4, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_ov_4 (overlap) = acch;
	numpat_ov_4 (overlap,:) = [numpat sum(numpat)];
	
    N = 4;
	[numpat, hdc_model_5] = hdctrain (L_SAMPL_DATA_5, SAMPL_DATA_5, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_5, TS_COMPLETE_5, hdc_model_5, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_ov_5 (overlap) = acch;
	numpat_ov_5 (overlap,:) = [numpat sum(numpat)];
end
 
for overlap = 1:1:100
    acc_hdc_ov_mean(overlap) = mean ([acc_hdc_ov_1(overlap) acc_hdc_ov_2(overlap) acc_hdc_ov_3(overlap) acc_hdc_ov_4(overlap) acc_hdc_ov_5(overlap)]);
    acc_hdc_ov_std(overlap) = std ([acc_hdc_ov_1(overlap) acc_hdc_ov_2(overlap) acc_hdc_ov_3(overlap) acc_hdc_ov_4(overlap) acc_hdc_ov_5(overlap)], 1);
end

% It produces Figure 9 in the paper
errorbar_groups (100*[acc_hdc_ov_mean(1) acc_hdc_ov_mean(1:5:100)], 100*[acc_hdc_ov_std(1) acc_hdc_ov_std(1:5:100)]);
ax = gca;
ax.YLabel.String = 'Accuracy (%)';
ax.YLim = [40 101];
ax.YTick = 40:5:101;
ax.XLabel.String = 'Number of gestures in a single window of classification';
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
ax.XLim = [0 22]; 
ax.XTick = 1:1:21;
ax.XTickLabel = 1.0:0.1:3;


%======================================================================================================================
%EXPLORE TRAINING SIZE; SVM HAS N=1 (to have its best accurcay)
overlap = 0;
for LF = 2:1:100 
    [L_SAMPL_DATA_1, SAMPL_DATA_1] = genTrainData (TS_COMPLETE_1, L_TS_COMPLETE_1, LF/100, '-------');
    [L_SAMPL_DATA_2, SAMPL_DATA_2] = genTrainData (TS_COMPLETE_2, L_TS_COMPLETE_2, LF/100, '-------');
    [L_SAMPL_DATA_3, SAMPL_DATA_3] = genTrainData (TS_COMPLETE_3, L_TS_COMPLETE_3, LF/100, '-------');
    [L_SAMPL_DATA_4, SAMPL_DATA_4] = genTrainData (TS_COMPLETE_4, L_TS_COMPLETE_4, LF/100, '-------');
    [L_SAMPL_DATA_5, SAMPL_DATA_5] = genTrainData (TS_COMPLETE_5, L_TS_COMPLETE_5, LF/100, '-------');

    N = 4;
    %N = 2;
	[numpat, hdc_model_1] = hdctrain (L_SAMPL_DATA_1, SAMPL_DATA_1, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_1, TS_COMPLETE_1, hdc_model_1, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_LF_1 (LF) = acch;
	numpat_LF_1 (LF,:) = [numpat sum(numpat)];
	
	svm_model_1 = svmtrain (L_SAMPL_DATA_1, SAMPL_DATA_1, ' -c 500 -h 0');
	num_sv_LF_1 (LF) = svm_model_1.totalSV;
	[v1, accs, v2] = svmpredict (L_TS_COMPLETE_1, TS_COMPLETE_1, svm_model_1);
	acc_svm_LF_1 (LF) = accs(1)/100;
	
    N = 4;
    %N = 2;
	[numpat, hdc_model_2] = hdctrain (L_SAMPL_DATA_2, SAMPL_DATA_2, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_2, TS_COMPLETE_2, hdc_model_2, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_LF_2 (LF) = acch;
	numpat_LF_2 (LF,:) = [numpat sum(numpat)];
    
    svm_model_2 = svmtrain (L_SAMPL_DATA_2, SAMPL_DATA_2, ' -c 500 -h 0');
	num_sv_LF_2 (LF) = svm_model_2.totalSV;
	[v1, accs, v2] = svmpredict (L_TS_COMPLETE_2, TS_COMPLETE_2, svm_model_2);
	acc_svm_LF_2 (LF) = accs(1)/100;
	
    N = 3;
    %N = 2;
	[numpat, hdc_model_3] = hdctrain (L_SAMPL_DATA_3, SAMPL_DATA_3, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_3, TS_COMPLETE_3, hdc_model_3, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_LF_3 (LF) = acch;
	numpat_LF_3 (LF,:) = [numpat sum(numpat)];
	
    svm_model_3 = svmtrain (L_SAMPL_DATA_3, SAMPL_DATA_3, ' -c 500 -h 0');
	num_sv_LF_3 (LF) = svm_model_3.totalSV;
	[v1, accs, v2] = svmpredict (L_TS_COMPLETE_3, TS_COMPLETE_3, svm_model_3);
	acc_svm_LF_3 (LF) = accs(1)/100;
	
    N = 5;
	[numpat, hdc_model_4] = hdctrain (L_SAMPL_DATA_4, SAMPL_DATA_4, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_4, TS_COMPLETE_4, hdc_model_4, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_LF_4 (LF) = acch;
	numpat_LF_4 (LF,:) = [numpat sum(numpat)];
	
    svm_model_4 = svmtrain (L_SAMPL_DATA_4, SAMPL_DATA_4, ' -c 500 -h 0');
	num_sv_LF_4 (LF) = svm_model_4.totalSV;
	[v1, accs, v2] = svmpredict (L_TS_COMPLETE_4, TS_COMPLETE_4, svm_model_4);
	acc_svm_LF_4 (LF) = accs(1)/100;
	
    N = 4;
	[numpat, hdc_model_5] = hdctrain (L_SAMPL_DATA_5, SAMPL_DATA_5, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_5, TS_COMPLETE_5, hdc_model_5, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_LF_5 (LF) = acch;
	numpat_LF_5 (LF,:) = [numpat sum(numpat)];
    
    svm_model_5 = svmtrain (L_SAMPL_DATA_5, SAMPL_DATA_5, ' -c 500 -h 0');
	num_sv_LF_5 (LF) = svm_model_5.totalSV;
	[v1, accs, v2] = svmpredict (L_TS_COMPLETE_5, TS_COMPLETE_5, svm_model_5);
	acc_svm_LF_5 (LF) = accs(1)/100;
end


for LF = 2:1:100    
    acc_hdc_LF_mean(LF) = mean([acc_hdc_LF_1(LF) acc_hdc_LF_2(LF) acc_hdc_LF_3(LF) acc_hdc_LF_4(LF) acc_hdc_LF_5(LF)]);  
    acc_hdc_LF_std(LF) = std([acc_hdc_LF_1(LF) acc_hdc_LF_2(LF) acc_hdc_LF_3(LF) acc_hdc_LF_4(LF) acc_hdc_LF_5(LF)]);  
    acc_svm_LF_mean(LF) = mean([acc_svm_LF_1(LF) acc_svm_LF_2(LF) acc_svm_LF_3(LF) acc_svm_LF_4(LF) acc_svm_LF_5(LF)]);  
    acc_svm_LF_std(LF) = std([acc_svm_LF_1(LF) acc_svm_LF_2(LF) acc_svm_LF_3(LF) acc_svm_LF_4(LF) acc_svm_LF_5(LF)]);  
    
    numpat_LF_mean(LF) = mean([numpat_LF_1(LF,6) numpat_LF_2(LF,6) numpat_LF_3(LF,6) numpat_LF_4(LF,6) numpat_LF_5(LF,6)]);  
    numpat_LF_std(LF) = std([numpat_LF_1(LF,6) numpat_LF_2(LF,6) numpat_LF_3(LF,6) numpat_LF_4(LF,6) numpat_LF_5(LF,6)]);  
    num_sv_LF_mean(LF) = mean([num_sv_LF_1(LF) num_sv_LF_2(LF) num_sv_LF_3(LF) num_sv_LF_4(LF) num_sv_LF_5(LF)]);  
    num_sv_LF_std(LF) = std([num_sv_LF_1(LF) num_sv_LF_2(LF) num_sv_LF_3(LF) num_sv_LF_4(LF) num_sv_LF_5(LF)]);  
end

% These produce the two plots in Figure 10
st = 10;
res = 5;
sp = 100;
errorbar_groups (100*[acc_svm_LF_mean(st:res:sp); acc_hdc_LF_mean(st:res:sp)], 100*[acc_svm_LF_std(st:res:sp); acc_hdc_LF_std(st:res:sp)])
ax = gca;
ax.YLabel.String = 'Accuracy (%)';
ax.YLim = [50 103];
ax.XLabel.String = 'Fraction of training (%)';
ax.XLim = [0 39];
ax.XTick = 1.5:2:39;
ax.XTickLabel = st:res:sp;
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
legend('SVM', 'HDC', 'location', 'best'); 

errorbar_groups ([num_sv_LF_mean(st:res:sp); numpat_LF_mean(st:res:sp)], [num_sv_LF_std(st:res:sp)-num_sv_LF_std(st:res:sp); numpat_LF_std(st:res:sp)-numpat_LF_std(st:res:sp)])
ax = gca;
ax.YLabel.String = 'Number';
ax.XLabel.String = 'Fraction of training (%)';
ax.XLim = [0 39];
ax.XTick = 1.5:2:39;
ax.XTickLabel = st:res:sp;
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
legend('Support vectors', 'Stored patterns in HDC', 'location', 'best');
%


%=====================================================================================================================
%EXPLORE DIMENSIONALITY
downSampRate = 250;
overlap = 0;
[TS_COMPLETE_1, L_TS_COMPLETE_1] = downSampling (COMPLETE_1, LABEL_1, downSampRate);
[TS_COMPLETE_2, L_TS_COMPLETE_2] = downSampling (COMPLETE_2, LABEL_2, downSampRate);
[TS_COMPLETE_3, L_TS_COMPLETE_3] = downSampling (COMPLETE_3, LABEL_3, downSampRate);
[TS_COMPLETE_4, L_TS_COMPLETE_4] = downSampling (COMPLETE_4, LABEL_4, downSampRate);
[L_SAMPL_DATA_1, SAMPL_DATA_1] = genTrainData (TS_COMPLETE_1, L_TS_COMPLETE_1, learningFrac, '-------');
[L_SAMPL_DATA_2, SAMPL_DATA_2] = genTrainData (TS_COMPLETE_2, L_TS_COMPLETE_2, learningFrac, '-------');
[L_SAMPL_DATA_3, SAMPL_DATA_3] = genTrainData (TS_COMPLETE_3, L_TS_COMPLETE_3, learningFrac, '-------');
[L_SAMPL_DATA_4, SAMPL_DATA_4] = genTrainData (TS_COMPLETE_4, L_TS_COMPLETE_4, learningFrac, '-------');

downSampRate = 50;
[TS_COMPLETE_5, L_TS_COMPLETE_5] = downSampling (COMPLETE_5, LABEL_5, downSampRate);
[L_SAMPL_DATA_5, SAMPL_DATA_5] = genTrainData (TS_COMPLETE_5, L_TS_COMPLETE_5, learningFrac, '-------');
cuttingAngle = 0.9;

for D = 10000:-100:100
	[iMch, chAM] = initItemMemories (D, 21);
    N = 4;
	[numpat, hdc_model_1] = hdctrain (L_SAMPL_DATA_1, SAMPL_DATA_1, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_1, TS_COMPLETE_1, hdc_model_1, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_D_1 (D) = acch;
	numpat_D_1 (D,:) = [numpat sum(numpat)];
	
    N = 4;
	[numpat, hdc_model_2] = hdctrain (L_SAMPL_DATA_2, SAMPL_DATA_2, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_2, TS_COMPLETE_2, hdc_model_2, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_D_2 (D) = acch;
	numpat_D_2 (D,:) = [numpat sum(numpat)];

    N = 3;
	[numpat, hdc_model_3] = hdctrain (L_SAMPL_DATA_3, SAMPL_DATA_3, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_3, TS_COMPLETE_3, hdc_model_3, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_D_3 (D) = acch;
	numpat_D_3 (D,:) = [numpat sum(numpat)];
	
    N = 5;
	[numpat, hdc_model_4] = hdctrain (L_SAMPL_DATA_4, SAMPL_DATA_4, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_4, TS_COMPLETE_4, hdc_model_4, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_D_4 (D) = acch;
	numpat_D_4 (D,:) = [numpat sum(numpat)];

    N = 4;
	[numpat, hdc_model_5] = hdctrain (L_SAMPL_DATA_5, SAMPL_DATA_5, iMch, chAM, D, N, percision, cuttingAngle);
	[acch] = test_slicing (L_TS_COMPLETE_5, TS_COMPLETE_5, hdc_model_5, iMch, chAM, D, N, percision, overlap/100);
	acc_hdc_D_5 (D) = acch;
	numpat_D_5 (D,:) = [numpat sum(numpat)];
end


for D = 10000:-100:100
    acc_hdc_D_mean(D) = mean ([acc_hdc_D_1(D) acc_hdc_D_2(D) acc_hdc_D_3(D) acc_hdc_D_4(D) acc_hdc_D_5(D)]);
    acc_hdc_D_std(D) = std ([acc_hdc_D_1(D) acc_hdc_D_2(D) acc_hdc_D_3(D) acc_hdc_D_4(D) acc_hdc_D_5(D)]);
end


M = acc_hdc_D_mean(100:100:1000);
M = [M acc_hdc_D_mean(2000)];
M = [M acc_hdc_D_mean(5000)];
M = [M acc_hdc_D_mean(10000)];
S = acc_hdc_D_std(100:100:1000);
S = [S acc_hdc_D_std(2000)];
S = [S acc_hdc_D_std(5000)];
S = [S acc_hdc_D_std(10000)];

M = acc_hdc_D_mean(100:100:1000);
M = [M acc_hdc_D_mean(2000:1000:10000)];
S = acc_hdc_D_std(100:100:1000);
S = [S acc_hdc_D_std(2000:1000:10000)];

% This produces a graph showing accuracy vs dimensions of hypervectors;
% some of its points are mentioned toward the end of Section V.B 
errorbar_groups(100*M, 100*S);
ax = gca;
ax.YLabel.String = 'Accuracy (%)';
ax.YLim = [40 101];
ax.XLabel.String = 'Dimension of hypervectors (in thousands)';
ax.FontWeight = 'Bold';
ax.FontName = 'Helvetica';
ax.XLim = [0 20]; %[0 14];
ax.XTick = 1:1:19;%1:1:13;
%ax.XTickLabel = {[100:100:900],'1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K' }; %{[100:100:1000], 2000, 5000, 10000};
ax.XTickLabel = {[0.1:0.1:1],[2:1:10] };
ax.YTick = 40:5:100;
%===============================================================================================================================
