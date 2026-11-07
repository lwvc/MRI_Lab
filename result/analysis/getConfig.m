function cfg = getConfig()

    % 路徑
    cfg.dataDir = 'C:\Lab\result\data';

    % 繪圖設定
    cfg.alpha = 0.45;
    cfg.MAD_YLIM = [0 12];

    % 模型設定：名稱、Excel、Sheet
    cfg.models = {
        'SFCN-reg T1',      'SFCN_reg.xlsx', 'T1WI'
        'SFCN-reg GM',      'SFCN_reg.xlsx', 'GMP'
        'SFCN-reg WM',      'SFCN_reg.xlsx', 'WMP'
        'SFCN-sm DGM',  'SFCN_sm_dgm.xlsx',  'dgm'
        'SFCN-sm T1',       'SFCN_sm.xlsx',  'T1WI'
        'SFCN-sm GM',       'SFCN_sm.xlsx',  'GMP'
        'SFCN-sm WM',       'SFCN_sm.xlsx',  'WMP'
        'SFCN-sm DGM',  'SFCN_sm_dgm.xlsx',  'DGM'

%         'ResNet T1',        'ResNet.xlsx',   'T1'
%         'ResNet T1 B0',     'ResNet.xlsx',   'T1_B0'
%         'ResNet T1 SEX',    'ResNet.xlsx',   'T1_SEX'
%         'ResNet T1 SEX B0', 'ResNet.xlsx',   'T1_SEX_B0'
    };

    % 模型顏色（依模型順序）
    cfg.MODEL_COLORS = [
%         0.00 0.45 0.00    % SFCN-reg T1
%         0.20 0.65 0.25    % SFCN-reg GM
%         0.55 0.80 0.52    % SFCN-reg WM
%         0.65 0.00 0.00    % SFCN-sm T1
%         0.88 0.25 0.25    % SFCN-sm GM
%         0.95 0.45 0.55    % SFCN-sm WM
%         0.02 0.00 0.64    % ResNet T1
%         0.30 0.35 0.80    % ResNet T1 SEX
%         0.25 0.45 0.85    % ResNet T1 B0
%         0.22 0.64 0.98    % ResNet T1 SEX B0
        0.65 0.18 0.20    % SFCN-sm T1
        0.87 0.40 0.30    % SFCN-sm GM
        0.90 0.61 0.26    % SFCN-sm WM
        0.71 0.36 0.64    % SFCN-sm DGM
    
        0.13 0.30 0.52    % SFCN-reg T1
        0.20 0.49 0.72    % SFCN-reg GM
        0.35 0.67 0.65    % SFCN-reg WM
        0.41 0.40 0.71    % SFCN-reg DGM
    ];

    % Maclaren 受試者顏色
    cfg.SUB_COLORS = [
        0.35 0.80 0.45    % Sub 1
        0.35 0.60 1.00    % Sub 2
        0.95 0.45 0.90    % Sub 3
    ];

    % 特殊受試者
    cfg.OAS1_OVERLAP = [
        "OAS1_0061"
        "OAS1_0092"
        "OAS1_0111"
        "OAS1_0150"
        "OAS1_0156"
        "OAS1_0202"
        "OAS1_0230"
        "OAS1_0353"
        "OAS1_0368"
        "OAS1_0379"
    ];

    cfg.MMRR_MOTION = "sub_742";
    cfg.MOTION_COLOR = [1.00 0.85 0.00];

end