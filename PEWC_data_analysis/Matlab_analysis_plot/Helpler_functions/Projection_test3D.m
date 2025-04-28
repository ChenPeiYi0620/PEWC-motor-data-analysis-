%% 主程式
% 定義四方體的八個頂點 (未排序)，例如一個單位立方體
vertices = [0 0 0;
            1 0 0;
            1 1 0;
            0 1 0;
            0 0 1;
            1 0 1;
            1 1 1;
            0 1 1];

% 定義任意三維點 P (例如在立方體上方)
P = [0.5, 2, 2];

% 計算 P 在四方體表面上的投影點
proj = projectPointOntoCube(vertices, P);

%% 繪圖顯示結果
figure;
hold on;
% 利用 convhulln 取得四方體的面（convhulln 會回傳三角形面）
K = convhulln(vertices);
trisurf(K, vertices(:,1), vertices(:,2), vertices(:,3), ...
        'FaceAlpha',0.3, 'EdgeColor','b', 'FaceColor',[0.8,0.8,1]);

% 繪製給定點 P 與投影點
plot3(P(1), P(2), P(3), 'ro', 'MarkerSize',10, 'LineWidth',2);
plot3(proj(1), proj(2), proj(3), 'gs', 'MarkerSize',10, 'LineWidth',2);
% 連線 P 與投影點
plot3([P(1) proj(1)], [P(2) proj(2)], [P(3) proj(3)], 'k--', 'LineWidth',2);

xlabel('X'); ylabel('Y'); zlabel('Z');
grid on; axis equal;
legend('四方體表面','給定點 P','投影點','連線','Location','Best');
title('給定點在四方體表面上的投影');
hold off;

%% 函數區段
function projPoint = projectPointOntoCube(vertices, P)
% projectPointOntoCube - 計算給定點 P 到由八個頂點構成的四方體表面上的最近投影點
%
% 輸入:
%   vertices - 8x3 陣列，表示四方體的頂點 (未排序，但必須構成凸多面體)
%   P        - 1x3 給定點 [x, y, z]
%
% 輸出:
%   projPoint - 最近投影點的座標 [x, y, z]
%
% 計算流程:
%   1. 利用 convhulln 找出多面體各三角形面。
%   2. 對每個三角形面，計算點到三角形的最短距離與對應投影點。
%   3. 回傳所有面中距離最小者作為投影點。

    % 取得 convex hull (每一列為一個三角形面，包含頂點的索引)
    K = convhulln(vertices);
    minDist = inf;
    projPoint = [];
    
    % 對每個三角形面進行計算
    for i = 1:size(K,1)
        % 取出該三角形的三個頂點
        A = vertices(K(i,1),:);
        B = vertices(K(i,2),:);
        C = vertices(K(i,3),:);
        % 計算 P 到三角形的最近點與距離
        [pt, dist] = pointToTriangle(P, A, B, C);
        if dist < minDist
            minDist = dist;
            projPoint = pt;
        end
    end
end

function [closestPoint, dist] = pointToTriangle(P, A, B, C)
% pointToTriangle - 計算點 P 到三角形 ABC 的最短距離與最近點
%
% 輸入:
%   P - 1x3 給定點
%   A, B, C - 1x3 三角形頂點
%
% 輸出:
%   closestPoint - 1x3 最近點在三角形上
%   dist         - 給定點與最近點之間的距離
%
% 演算法參考自常見點到三角形距離計算方法

    % 向量計算
    AB = B - A;
    AC = C - A;
    AP = P - A;
    d1 = dot(AB, AP);
    d2 = dot(AC, AP);
    if d1 <= 0 && d2 <= 0
        closestPoint = A;
        dist = norm(P - A);
        return;
    end
    
    BP = P - B;
    d3 = dot(AB, BP);
    d4 = dot(AC, BP);
    if d3 >= 0 && d4 <= d3
        closestPoint = B;
        dist = norm(P - B);
        return;
    end
    
    CP = P - C;
    d5 = dot(AB, CP);
    d6 = dot(AC, CP);
    if d6 >= 0 && d5 <= d6
        closestPoint = C;
        dist = norm(P - C);
        return;
    end
    
    % 檢查邊 AB 的區域
    vc = d1 * d4 - d3 * d2;
    if vc <= 0 && d1 >= 0 && d3 <= 0
        v = d1 / (d1 - d3);
        closestPoint = A + v * AB;
        dist = norm(P - closestPoint);
        return;
    end
    
    % 檢查邊 AC 的區域
    vb = d2 * d5 - d1 * d6;
    if vb <= 0 && d2 >= 0 && d6 <= 0
        w = d2 / (d2 - d6);
        closestPoint = A + w * AC;
        dist = norm(P - closestPoint);
        return;
    end
    
    % 檢查邊 BC 的區域
    va = d3 * d6 - d4 * d5;
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closestPoint = B + w * (C - B);
        dist = norm(P - closestPoint);
        return;
    end
    
    % 如果以上都不成立，則 P 在三角形內部對應的投影即為最近點
    denom = 1 / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;
    closestPoint = A + AB * v + AC * w;
    dist = norm(P - closestPoint);
end
