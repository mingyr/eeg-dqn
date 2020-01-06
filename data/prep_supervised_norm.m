%% layout of some data structure, only for demo
% events = [249  252  253  249  231  251  253  249  251]
%            1    2    3    4    5    6    7    8    9
% dev_indices = [ 2    3              6    7         9 ]
% dev_inx     = [ 1    2              3    4         5 ]
clc;
clear;
warning off;

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;


% we regard reaction time less than 0.5 is invalid
%% Set data path

datapath = fullfile('D:\\Share\\lane-keeping');
savepath = fullfile('D:\\EEG-RL\\RT-norm\\sr128-supervised');

% listname = dir(datapath);
% listname = listname(~ismember({listname.name}, {'.', '..'}));
% listname = {listname.name};

listname = {
%     's01_060227n'
%     's01_060926_1n'
%     's01_060926_2n'
%     's01_061102n'

%     
    's09_060313n'
    's09_060317n'
    's09_060720_1n'

    's41_061225n'
    's41_080530n'
    's41_091104n'

%     's49_080522n'
%     's49_080527n'
% 
%     's53_081018n'
%     's53_090918n'
% 
%     's48_080501n'
% 
%     's50_080725n'
% 
%     's52_081017n'
};

sr = 128; % sampling rate
unit = 3; % length of baseline, in the unit of second
offset = sr * unit - 1;

rt_min = 0.5;
rt_max = 8;

for dn = 1 : numel(listname)
    filepath = fullfile(datapath, listname{dn});
    [~, name, ~] = fileparts(listname{dn});
    filename = [name, '.set'];

    % load the data
    EEG = pop_loadset('filename', filename, 'filepath', filepath);
    EEG = eeg_checkset(EEG);

    % select corresponding channels
    channels = {EEG.chanlocs.labels};
    channels = channels(~ismember(channels, {'A1', 'A2', 'vehicle position'}));

    EEG = pop_select(EEG, 'channel', channels);
    EEG = eeg_checkset( EEG );
    
    if EEG.srate == 500
        EEG = pop_eegfiltnew(EEG, 0.5,50,3300,0,[],0);
    elseif EEG.srate == 1000
        EEG = pop_eegfiltnew(EEG, 0.5,50,6600,0,[],0);
    else
        disp('unexpected sampling rate')
    end
    
    % assert((sr == EEG.srate), 'unexpected sampling rate')    
    EEG = pop_resample(EEG, sr);
    EEG = eeg_checkset( EEG );

    events = [EEG.event.type];
    dev_indices = find(events == 251 | events == 252);

    RT = [];
    data = [];
    rt_trace = [];
    count = 1;
    
    for dev_idx = 1 : numel(dev_indices)
        rt = 0;
        if (dev_indices(dev_idx) + 1) <= numel(events) && ...
                events(dev_indices(dev_idx) + 1) == 253
            epoch_start = EEG.event(dev_indices(dev_idx)).latency;            
            epoch_end = EEG.event(dev_indices(dev_idx) + 1).latency;
            rt = (epoch_end - epoch_start) / sr;
            rt = max(rt, rt_min);
            rt = min(rt, rt_max);
       
            RT(count) = rt;            
            data(:, :, count) = EEG.data(:, (epoch_start - offset) : epoch_start);
            rt_trace(count) = epoch_start;
            count = count + 1;
        end    
    end

    RT_ = [];
    for i = 1:numel(RT)
        RT_(i) = RT(i);            
        rt_means = [RT(i)];
        if RT(i) > 0
            j = 1;
            loop_valid = 1;
            while (i - j > 0) && loop_valid > 0
                if rt_trace(i - j) > 0
                    if abs((rt_trace(i - j) - rt_trace(i)) / sr) <= 45
                        assert(RT(i - j) > 0)
                        rt_means = [RT(i - j) rt_means];
                        j = j + 1;
                    else
                        loop_valid = 0;
                    end
                else
                    j = j + 1;
                end
            end
                        
            j = 1;
            loop_valid = 1;
            while (i + j <= numel(RT)) && loop_valid > 0
                if rt_trace(i + j) > 0
                    if abs((rt_trace(i + j) - rt_trace(i)) / sr) <= 45
                        assert(RT(i + j) > 0)
                        rt_means = [rt_means RT(i + j)];
                        j = j + 1;
                    else
                        loop_valid = 0;
                    end
                else
                    j = j + 1;
                end
            end
            
            disp(i);
            disp(RT(i));
            disp(rt_means);
    
        end
        if numel(rt_means) > 0
            RT_(i) = mean(rt_means);
        end
    end
    
    RT = RT_;    
    
    filepath = fullfile(savepath, [name, '.mat']);
    save(filepath, 'data', 'RT');

end
