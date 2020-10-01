% % Author: Qingzhe  
% % Created Date: 05\\01\\2020
%%
clear;clc;
% batchIdx = 1
subjects = dlmread('subjects.txt','|');
types = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"];

%  for subjectIdx =  1:400 %(batchIdx-1)*batchSize+1:batchIdx*batchSize
for subjectIdx =  110:400 %(batchIdx-1)*batchSize+1:batchIdx*batchSize

%     if ~exist(strcat('FC_',task), 'dir')
%            mkdir(strcat('FC_',task));
%     end
    %%
    WBC = 'C:/workbench//bin_windows64/wb_command.exe';

    % % batchSize = 357
    fid = fopen('batch.bat');
    %%
   
    for task = types
        %  7z x F:\TFMRI_401_800\178142_3T_tfMRI_EMOTION_preproc.zip -oG:\
        try  
            % unzip the downloaded file recursively
        subject = subjects(subjectIdx);
%         -aoa	Overwrite All existing files without prompt.
%         -aos	Skip extracting of existing files.
%         7z x -aos H:\TFMRI_1_400\100206_3T_Structural_preproc.zip -oG:\HCP
        unzip_structalFile = strcat('7z x -aos H:\TFMRI_1_400\',num2str(subject),'_3T_Structural_preproc.zip -oG:\HCP');
        exit = system(unzip_structalFile)
        if exit~=0
                error('#######  failed to extract structural files!')
        end
        unzip = strcat('7z x -aos H:\TFMRI_1_400\',num2str(subject),'_3T_tfMRI_',task,'_preproc.zip -oG:HCP *Atlas.dtseries.nii -r')
        exit = system(unzip)
        if exit~=0
                error('failed to extract files!')
        end
        sprintf('%d: %d',subjectIdx, subject)
        path = sprintf('G:/HCP/%d/MNINonLinear/',subject);
        % G:\177746\MNINonLinear\Results\tfMRI_EMOTION_RL
%         tf_path = sprintf('E:/%d/MNINonLinear/',subject);
        tf_path = sprintf('G:/HCP/%d/MNINonLinear/',subject);

% %    %     tsROI = zeros(4800,68);
% %     %     tsROI = zeros(0,68);
% %     %     tsROI= [];
        tsId = 1;
        tsLen = 0;
        keyIdx = 0;
        for key = 1:35
            % ROI 4 does not have SC and disgarded from the original 70 parcellations.
            if key == 4 
                continue; 
            end
            keyIdx = keyIdx+1;
            step1 = sprintf('%s -gifti-label-to-roi %sfsaverage_LR32k/%d.L.aparc.32k_fs_LR.label.gii %sROIs/L%d.func.gii -key %d',WBC,path,subject,path,key,key);
            exit =system(step1);

            if exit~=0
                error('failed: file incomplete')
            end
            step2 = sprintf('%s -cifti-roi-average %sResults/tfMRI_%s_LR/tfMRI_%s_LR_Atlas.dtseries.nii C:\\Users\\qli10\\temp_avgTS_1_400.txt -left-roi %sROIs/L%d.func.gii ',WBC,tf_path,task,task,path,key);
            exit =system(step2);
            if exit~=0
                error('#### failed: file incomplete')
            end
            ts = dlmread('C:\\Users\\qli10\\temp_avgTS_1_400.txt');
            if key==1
                tsLen = length(ts);
                tsROI=zeros(tsLen*2,68);
    %         else
    %             ts = [ts  dlmread('C:\\Users\\qli10\\temp_avgTS_1_400.txt')];
            end
            tsROI(1:tsLen,keyIdx) = ts;
            step2 = sprintf('%s -cifti-roi-average  %sResults/tfMRI_%s_RL/tfMRI_%s_RL_Atlas.dtseries.nii C:\\Users\\qli10\\temp_avgTS_1_400.txt -left-roi %sROIs/L%d.func.gii ',WBC,tf_path,task,task,path,key);
            exit =system(step2);
            if exit~=0
                error('failed: file incomplete')
            end
            tsROI(tsLen+1:2*tsLen,keyIdx) =  dlmread('C:\\Users\\qli10\\temp_avgTS_1_400.txt');
%             step2 = sprintf('%s -cifti-roi-average %sResults/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii C:\\Users\\qli10\\temp_avgTS_1_400.txt -left-roi %sROIs/L%d.func.gii ',WBC,path,path,key);
%             exit =system(step2);
%             if exit~=0
%                 error('failed: file incomplete')
%             end
%             tsROI(2*tsLen+1:3*tsLen,keyIdx) =  dlmread('C:\\Users\\qli10\\temp_avgTS_1_400.txt');
%             step2 = sprintf('%s -cifti-roi-average %sResults/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii C:\\Users\\qli10\\temp_avgTS_1_400.txt -left-roi %sROIs/L%d.func.gii ',WBC,path,path,key);
%             exit =system(step2);
%             if exit~=0
%                 error('failed: file incomplete')
%             end
%             tsROI(3*tsLen+1:4*tsLen,keyIdx) =  dlmread('C:\\Users\\qli10\\temp_avgTS_1_400.txt');
            tsId = tsId+1;
        end
        for key = 1:35
            if key == 4 
                continue; 
            end
            keyIdx = keyIdx+1;
        step1 = sprintf('%s -gifti-label-to-roi %sfsaverage_LR32k/%d.R.aparc.32k_fs_LR.label.gii %sROIs/R%d.func.gii -key %d',WBC,path,subject,path,key,key);
        exit = system(step1);
        if exit~=0
            error('######## failed on Step 1: file incomplete')
        end
        step2 = sprintf('%s -cifti-roi-average %sResults/tfMRI_%s_LR/tfMRI_%s_LR_Atlas.dtseries.nii C:\\Users\\qli10\\temp_avgTS_1_400.txt -right-roi %sROIs/R%d.func.gii ',WBC,tf_path,task,task,path,key);
        exit =system(step2);

        if exit~=0
            error('######## failed on LR step 2: file incomplete')
        end
        tsROI(1:tsLen,keyIdx) =  dlmread('C:\\Users\\qli10\\temp_avgTS_1_400.txt');
        step2 = sprintf('%s -cifti-roi-average %sResults/tfMRI_%s_RL/tfMRI_%s_RL_Atlas.dtseries.nii C:\\Users\\qli10\\temp_avgTS_1_400.txt -right-roi %sROIs/R%d.func.gii',WBC,tf_path,task,task,path,key);
        exit =system(step2);
        if exit~=0
            error('######## failed on RL step 2: file incomplete')
        end
        tsROI(tsLen+1:2*tsLen,keyIdx) =  dlmread('C:\\Users\\qli10\\temp_avgTS_1_400.txt');

        tsId = tsId+1;
        end

        
        %%     
        correlationFC=nets_netmats(tsROI,0,'corr');
        partialCorrelationFC_L1=nets_netmats(tsROI,0,'icov',1);
        partialCorrelationFC_L2=nets_netmats(tsROI,0,'ridgep',0.01);
        %  heatmap(netmat2,'Colormap',jet);
%         dir = sprintf('FC_%s/%d',task,subject);
         dir = 'FC_tfMRI_0823'
          if ~exist(dir, 'dir')
               mkdir(dir);
          end
         fileName = sprintf('%s/%d_%s.mat',dir,subject,task)
         save(fileName,'correlationFC','partialCorrelationFC_L1','partialCorrelationFC_L2','tsROI','-v7.3');
        
%%

       catch 
            sprintf('%s: Subject %d Failed',task,subject) 
            dlmwrite(strcat('incompleteSubjects_',task,'_0823.txt'),subject,'-append','precision','%d')

       end
    end
end