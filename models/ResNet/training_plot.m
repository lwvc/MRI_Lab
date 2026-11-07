clear; clc; close all;

% 讀取 training pool
file = "C:\Lab\model_val_test_train\ResNet\train_pool.csv";
T = readtable(file, "VariableNamingRule","preserve");

id      = string(T.Subject_ID);
sex     = T.("SEX(0=Male,1=Female)");
age     = T.AGE;
scanner = T.Scanner;

% 判斷資料集
dataset = strings(height(T),1);

dataset(startsWith(id,"ABIDE1_")) = "ABIDE1";
dataset(startsWith(id,"ABIDE2_")) = "ABIDE2";
dataset(startsWith(id,"FCON_"))   = "FCP1000";
dataset(startsWith(id,"NKI-RS_")) = "NKI-RS";
dataset(startsWith(id,"IXI_"))    = "IXI";
dataset(startsWith(id,"OAS1_"))   = "OAS1";
dataset(startsWith(id,"HCA"))     = "HCP-Aging";

order = ["ABIDE1","ABIDE2","FCP1000","NKI-RS", ...
         "IXI","OAS1","HCP-Aging"];

% 檢查是否有無法辨識的 Subject ID
unknown = id(dataset == "");

if ~isempty(unknown)
    warning("有 %d 筆 Subject_ID 無法判斷資料庫。", numel(unknown));
    disp(unknown);
end

% 移除異常資料
valid = dataset ~= "" & ismember(sex,[0 1]) & ~isnan(age);

dataset = dataset(valid);
sex     = sex(valid);
age     = age(valid);
scanner = scanner(valid);

% 顏色
maleColor   = [174 216 232] / 255;   % #AED8E8
femaleColor = [244 197 207] / 255;   % #F4C5CF

% Violin plot
figure("Color","w","Position",[100 100 1000 650]);
hold on;

spacing = 0.75;
xpos = (1:numel(order)) * spacing;

width = 0.30;

for i = 1:numel(order)

    xc = xpos(i);

    for s = 0:1

        a = age(dataset == order(i) & sex == s);

        if isempty(a)
            continue;
        end

        [f,y] = ksdensity(a,"Support",[0 105]);
        f = f / max(f) * width;

        if s == 0
            % Male：左半邊
            x = xc - f;
            c = maleColor;
        else
            % Female：右半邊
            x = xc + f;
            c = femaleColor;
        end

        patch( ...
            [xc*ones(size(y)) fliplr(x)], ...
            [y fliplr(y)], ...
            c, ...
            "FaceAlpha",0.9, ...
            "EdgeColor",[0.3 0.3 0.3]);

        % Q1 / Median / Q3
        q = prctile(a,[25 50 75]);

        if s == 0
            xr = [xc-0.15 xc];
        else
            xr = [xc xc+0.15];
        end

        for k = 1:3
            plot(xr,[q(k) q(k)],"k--","LineWidth",1);
        end
    end
end

% 圖表設定
xlim([xpos(1)-0.45 xpos(end)+0.45]);
ylim([0 105]);

xticks(xpos);
xticklabels(order);
yticks(0:10:100);

xlabel("Dataset");
ylabel("Age");
title("Age Distribution of Training Pool (n = 3471)");

box on;
grid on;

h1 = patch(nan,nan,maleColor);
h2 = patch(nan,nan,femaleColor);

legend([h1 h2],["Male","Female"], ...
       "Location","northeast");

% Print 統計表
fprintf("\n%-12s %-7s %-14s %-14s %-25s\n", ...
    "資料庫","人數","年齡","性別(M/F)","主磁場");

fprintf("%s\n",repmat('-',1,80));

total = 0;

for i = 1:numel(order)

    idx = dataset == order(i);

    N = sum(idx);
    M = sum(idx & sex == 0);
    F = sum(idx & sex == 1);

    ageText = sprintf("%.1f - %.1f", ...
        min(age(idx)),max(age(idx)));

    sexText = sprintf("%d / %d",M,F);

    n15 = sum(idx & scanner == 1.5);
    n30 = sum(idx & scanner == 3);

    if n15 > 0 && n30 > 0
        fieldText = sprintf("1.5T:%d / 3T:%d",n15,n30);

    elseif n15 > 0
        fieldText = sprintf("1.5T:%d",n15);

    elseif n30 > 0
        fieldText = sprintf("3T:%d",n30);

    else
        fieldText = "Unknown";
    end

    fprintf("%-12s %-7d %-14s %-14s %-25s\n", ...
        order(i),N,ageText,sexText,fieldText);

    total = total + N;
end

fprintf("%s\n",repmat('-',1,80));
fprintf("Total: %d 人\n",total);