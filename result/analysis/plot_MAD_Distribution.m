clear; clc; close all;

cfg = getConfig();

outDir = 'C:\Lab\result\figures';

if ~isfolder(outDir)
    mkdir(outDir);
end

datasets = ["Maclaren", "OAS1", "MMRR"];
nModels = size(cfg.models, 1);

for d = 1:numel(datasets)

    db = datasets(d);

    figure('Color', 'w', 'Position', [100 100 1350 620]);
    hold on;

    summaryMAD = nan(nModels,1);
    numPairs = zeros(nModels,1);

    % 各模型 MAD
    for m = 1:nModels

        data = loadDatasetData(cfg, db, m);

        y = data.pairDiff;
        subs = data.pairSub;

        % OAS1 ResNet：排除已見過的 10 位受試者
        if db == "OAS1" && contains(cfg.models{m,1}, 'ResNet')
            keep = ~ismember(subs, cfg.OAS1_OVERLAP);
            y = y(keep);
            subs = subs(keep);
        end

        numPairs(m) = numel(y);
        summaryMAD(m) = mean(y);

        % 繪製資料點
        if db == "Maclaren"

            % 三位受試者固定左、中、右
            for s = 1:3

                sub = sprintf('sub-%02d', s);
                idx = subs == sub;

                n = sum(idx);
                x = m + (s-2)*0.15 + linspace(-0.025,0.025,n)';

                scatter(x, y(idx), 45, 'o', 'filled', ...
                    'MarkerFaceColor', cfg.SUB_COLORS(s,:), ...
                    'MarkerFaceAlpha', cfg.alpha, ...
                    'MarkerEdgeColor', 'none');
            end

        else

            % OAS1 / MMRR 橫向分散
            n = numel(y);
            x = m + linspace(-0.16,0.16,n)';

            isMotion = db == "MMRR" & subs == cfg.MMRR_MOTION;

            % 一般資料點
            scatter(x(~isMotion), y(~isMotion), 45, 'o', 'filled', ...
                'MarkerFaceColor', cfg.MODEL_COLORS(m,:), ...
                'MarkerFaceAlpha', cfg.alpha, ...
                'MarkerEdgeColor', 'none');

            % MMRR sub_742：黃色星號
            if any(isMotion)
                scatter(x(isMotion), y(isMotion), 130, 'p', 'filled', ...
                    'MarkerFaceColor', cfg.MOTION_COLOR, ...
                    'MarkerFaceAlpha', cfg.alpha, ...
                    'MarkerEdgeColor', [0.25 0.25 0.25], ...
                    'LineWidth', 1);
            end

        end

        % MAD 平均值：灰色虛線
        plot([m-0.25 m+0.25], ...
             [summaryMAD(m) summaryMAD(m)], '--', ...
             'Color', [0.4 0.4 0.4], 'LineWidth', 2);

        text(m, summaryMAD(m)+0.45, ...
            sprintf('%.2f', summaryMAD(m)), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 10, 'FontWeight', 'bold', ...
            'Color', [0.3 0.3 0.3]);

    end

    % 圖形設定
    xlim([0.4 nModels+0.6]);
    xticks(1:nModels);
    xticklabels(cfg.models(:,1));
    xtickangle(35);

    ylim(cfg.MAD_YLIM);
    yticks(0:2:12);

    ylabel('|Scan 1 - Scan 2| (years)', 'FontSize', 12);
    title(sprintf('MAD - %s', db), 'FontSize', 14);

    grid on;
    box on;
    set(gca, 'FontSize', 10, 'TickDir', 'out');

    % 圖例
    hMAD = plot(nan,nan,'--', ...
        'Color', [0.4 0.4 0.4], 'LineWidth', 2);

    if db == "Maclaren"

        hSub = gobjects(1,3);

        for s = 1:3
            hSub(s) = scatter(nan,nan,50,'o','filled', ...
                'MarkerFaceColor', cfg.SUB_COLORS(s,:), ...
                'MarkerFaceAlpha', cfg.alpha, ...
                'MarkerEdgeColor', 'none');
        end

        legend([hSub hMAD], ...
            {'Subject 1','Subject 2','Subject 3','MAD'}, ...
            'Location', 'northeast');

    elseif db == "MMRR"

        hMotion = scatter(nan,nan,120,'p','filled', ...
            'MarkerFaceColor', cfg.MOTION_COLOR, ...
            'MarkerFaceAlpha', cfg.alpha, ...
            'MarkerEdgeColor', [0.25 0.25 0.25]);

        legend([hMAD hMotion], ...
            {'MAD','sub_742 (motion)'}, ...
            'Location', 'northeast', ...
            'Interpreter', 'none');

    else

        legend(hMAD, 'MAD', 'Location', 'northeast');

    end



    % 命令列輸出
    fprintf('\n========== %s MAD Summary ==========\n', db);

    T_summary = table( ...
        string(cfg.models(:,1)), numPairs, round(summaryMAD, 2), ...
        'VariableNames', {'Model','N_Pairs','MAD_yr'});

    disp(T_summary);


    drawnow;
    fileBase = fullfile(outDir, sprintf('MAD_%s', db));

    exportgraphics(gcf, [fileBase '.png'], 'Resolution', 300);
    savefig(gcf, [fileBase '.fig']);

end