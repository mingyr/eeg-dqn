% clc;
% clear;
warning off;

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;


%{
% we regard reaction time less than 0.5 is invalid

excited: 0.5 - 1.0   - 1 - diff: 1    medium: 0.75
alert: 1.0 - 2.5   - 2 - diff: 1.5  medium: 1.75
drowsy: 2.5 - 4.5  - 3 - diff: 2    medium: 3.5
sleepy: 4.5 - +inf - 4 - diff: 3    medium: 6

actions: excited -> alert   (+1)
         alert   -> drowsy  (+1)
         drowsy  -> sleepy  (+1)
         sleep   -> alert   (-2)
         sleep   -> drowsy  (-1)
%}
%% Set data path

datapath = fullfile('D:\\Share\\lane-keeping');
savepath = fullfile('D:\\EEG-RL\\RT-norm\\sr128');

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

rt_min = 0.5;
rt_max = 8;

tau0 = 0.3; % threshold for drowsiness, 0.3s is estimated the least RT

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
    rt_trace = [];
    data = [];

    count = 1;
    dev_idx = 1;
    
    while 1
        % decide the indices
        indices = (count - 1) * unit * sr + 1 : count * unit * sr;
        if indices(end) > size(EEG.data, 2)
            break;
        end

        % save corresponding data
        data(:, :, count) = EEG.data(:, indices);
       
        rt = 0;
        epoch_start = 0;
        % check whether there is an deviation event in the indices
        % make sure evt_idx is valid
        if dev_idx <= numel(dev_indices)
            % test whether the position of dev_events is contained in indices 
            if ismember(round(EEG.event(dev_indices(dev_idx)).latency), indices)
                % make sure this is not the last event
                if (dev_indices(dev_idx) + 1) <= numel(events)
                    % make sure the following event is a response onset
                    if events(dev_indices(dev_idx) + 1) == 253
                        epoch_start = EEG.event(dev_indices(dev_idx)).latency;
                        epoch_end = EEG.event(dev_indices(dev_idx) + 1).latency;
                        rt = (epoch_end - epoch_start) / sr;
                        rt = max(rt, rt_min);
                        rt = min(rt, rt_max);
                    end
                end
                
                dev_idx = dev_idx + 1;
            end
        end
        
        RT(count) = rt;
        rt_trace(count) = epoch_start;
        
        count = count + 1;
    end
    
    % averaging
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
