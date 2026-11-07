function data = loadDatasetData(cfg, datasetName, modelIdx)
    % 讀取 Excel
    filePath = fullfile(cfg.dataDir, cfg.models{modelIdx, 2});
    sheetName = sprintf('%s_%s', datasetName, cfg.models{modelIdx, 3});
    
%     T = readtable(filePath, 'Sheet', sheetName, 'VariableNamingRule', 'preserve');
    T = readtable(filePath, 'Sheet', sheetName, ...
    'Range', 'A1', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve');

    % 四個欄位
    data.subID   = string(T.("Subject ID"));
    data.trueAge = double(T.("True Age"));
    data.predAge = double(T.("Predicted Age"));
    data.pairID  = string(T.("pair"));

    % 檢查配對欄位
    if any(ismissing(data.pairID) | strlength(strtrim(data.pairID)) == 0)
        error('Excel 的 pair 欄位存在空值：%s', sheetName);
    end
    
    % 依 pair 計算兩次掃描差異
    [pairs, ~, group] = unique(data.pairID);

    pairDiff = [];
    pairSub  = strings(0,1);
    
    for p = 1:numel(pairs)
        rows = find(group == p);

        if numel(rows) ~= 2
            continue;
        end

        diffAge = abs(data.predAge(rows(1)) - data.predAge(rows(2)));

        % 根據 Excel 的 pair 格式取得受試者 ID
        switch string(datasetName)
            case "Maclaren"
                sub = extractBefore(data.subID(rows(1)), "_run-");

            case "OAS1"
                sub = pairs(p);

            case "MMRR"
                sub = "sub_" + pairs(p);
        end

        pairDiff(end+1,1) = diffAge;
        pairSub(end+1,1)  = sub;

    end

    data.pairDiff = pairDiff;
    data.pairSub  = pairSub;

end