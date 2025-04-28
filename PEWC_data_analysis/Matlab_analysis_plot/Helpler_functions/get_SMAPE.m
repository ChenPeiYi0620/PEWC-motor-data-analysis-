function smape_value=get_SMAPE (actual, predicted)

% 函數描述：
    % 計算兩個矩陣之間的 SMAPE
    % actual 和 predicted 必須為相同大小的矩陣
    
    % 檢查矩陣大小是否一致
    if size(actual) ~= size(predicted)
        error('actual 和 predicted 必須具有相同的大小');
    end
    
    % 計算 SMAPE
    numerator = abs(actual - predicted);       % 絕對誤差
    denominator = (abs(actual) + abs(predicted)) / 2; % 平均值作為分母
    smape_matrix = numerator ./ denominator;  % 每個元素的 SMAPE 值
    
    % 處理分母為零的情況，避免 NaN
    smape_matrix(denominator == 0) = 0; % 如果分母為零，該位置的 SMAPE 設為 0
    
    % 平均所有元素，轉換為百分比
    smape_value = mean(smape_matrix(:)) * 100;


end 