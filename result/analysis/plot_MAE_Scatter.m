clear; clc; close all;

cfg = getConfig();

outDir = 'C:\Lab\result\figures';

if ~isfolder(outDir)
    mkdir(outDir);
end

datasets = ["Maclaren", "OAS1", "MMRR"];
nModels = size(cfg.models, 1);

regColor = [0.72 0.68 0.40];
specialColor = [1.00 0.01 0.1]; 

for d = 1:numel(datasets)

    db = datasets(d);

    % 統一該資料集的座標範圍
    minAge = inf;
    maxAge = -inf;

    for m = 1:nModels
        dt = loadDatasetData(cfg, db, m);

        minAge = min([minAge; dt.trueAge; dt.predAge], [], 'omitnan');
        maxAge = max([maxAge; dt.trueAge; dt.predAge], [], 'omitnan');
    end

    axisLim = [minAge-1, maxAge+1];

    % 建立長方形圖
    figure('Color', 'w', 'Position', [100 100 1600 850]);
%    改 tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
    tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    summaryMAE = nan(nModels,1);
    numSamples = zeros(nModels,1);

    for m = 1:nModels

        nexttile;
        hold on;

        dt = loadDatasetData(cfg, db, m);

        % 有效資料與 MAE
        valid = isfinite(dt.trueAge) & isfinite(dt.predAge);

        trueAge = dt.trueAge(valid);
        predAge = dt.predAge(valid);
        subID = dt.subID(valid);

        numSamples(m) = numel(trueAge);
        summaryMAE(m) = mean(abs(predAge - trueAge));

        % 繪製資料點
        if db == "Maclaren"

            for s = 1:3
                idx = startsWith(subID, sprintf('sub-%02d', s));

                scatter(trueAge(idx), predAge(idx), 35, 'o', 'filled', ...
                    'MarkerFaceColor', cfg.SUB_COLORS(s,:), ...
                    'MarkerFaceAlpha', cfg.alpha, ...
                    'MarkerEdgeColor', 'none');
            end

        else

            isSpecial = false(size(subID));

            if db == "MMRR"

                isSpecial = startsWith(subID, ["sub_502_", "sub_742_"]);

            elseif db == "OAS1" && contains(cfg.models{m,1}, 'ResNet')

                cleanID = regexprep(subID, '_MR\d+$', '');
                isSpecial = ismember(cleanID, cfg.OAS1_OVERLAP);

            end

            % 一般資料點
            scatter(trueAge(~isSpecial), predAge(~isSpecial), ...
                35, 'o', 'filled', ...
                'MarkerFaceColor', cfg.MODEL_COLORS(m,:), ...
                'MarkerFaceAlpha', cfg.alpha, ...
                'MarkerEdgeColor', 'none');

            % 特殊受試者：紅點
            if any(isSpecial)
                scatter(trueAge(isSpecial), predAge(isSpecial), ...
                    45, 'o', 'filled', ...
                    'MarkerFaceColor', specialColor, ...
                    'MarkerFaceAlpha', 0.6, ...
                    'MarkerEdgeColor', 'none');
            end

        end

        % y = x 參考線
        plot(axisLim, axisLim, '--', ...
            'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);

        % 只有 MMRR 畫回歸線
        if db == "MMRR" && numel(trueAge) >= 2

            p = polyfit(trueAge, predAge, 1);

            plot(axisLim, polyval(p, axisLim), '-', ...
                'Color', regColor, ...
                'LineWidth', 1.5);
        end

        % 座標與網格
        xlim(axisLim);
        ylim(axisLim);

        gridTicks = ceil(axisLim(1)/5)*5 : 5 : floor(axisLim(2)/5)*5;

        xticks(gridTicks);
        yticks(gridTicks);

        grid on;
        box on;

        title(sprintf('%s (MAE: %.2f, N=%d)', ...
            cfg.models{m,1}, summaryMAE(m), numSamples(m)), ...
            'FontSize', 10);

        xlabel('True Age (yr)', 'FontSize', 9);
        ylabel('Predicted Age (yr)', 'FontSize', 9);

        % 不使用 axis square，適合 PPT 長方形排版

    end

    % 命令列輸出
    fprintf('\n========== %s MAE Summary ==========\n', db);

    T_summary = table( ...
        string(cfg.models(:,1)), numSamples, summaryMAE, ...
        'VariableNames', {'Model', 'N', 'MAE_yr'});

    disp(T_summary);


    drawnow;
    fileBase = fullfile(outDir, sprintf('MAE_%s', db));

    exportgraphics(gcf, [fileBase '.png'], 'Resolution', 300);
    savefig(gcf, [fileBase '.fig']);

end