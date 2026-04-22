% plot_multi_circuit.m  -  2 scenarios: Figure-8 (moderate) + Vertical Loop (acrobatic)
% Plots PMM trajectories colored by speed, with shared colorbar.
% Run from:  path_planing/

close all; clc;

%% -- Load PMM results ---------------------------------------------------------

S(1) = load('path_PMM_results_fig8.mat');
S(2) = load('path_PMM_results_loop.mat');

titles = {
    sprintf('(a) Figure-8  |  T_{opt}=%.2f s,  v_{max}=%.2f m/s', ...
            S(1).T_opt, max(sqrt(S(1).vx.^2 + S(1).vy.^2 + S(1).vz.^2)));
    sprintf('(b) Vertical Loop  |  T_{opt}=%.2f s,  v_{max}=%.2f m/s', ...
            S(2).T_opt, max(sqrt(S(2).vx.^2 + S(2).vy.^2 + S(2).vz.^2)));
};

%% -- Shared colorbar range (max over both) ------------------------------------

v_all_max = max([ ...
    max(sqrt(S(1).vx.^2 + S(1).vy.^2 + S(1).vz.^2)), ...
    max(sqrt(S(2).vx.^2 + S(2).vy.^2 + S(2).vz.^2))  ...
]);
v_all_max = ceil(v_all_max);          % round up for clean colorbar

%% -- Shared workspace ---------------------------------------------------------

all_pos = [ ...
    S(1).px(:), S(1).py(:), S(1).pz(:); ...
    S(2).px(:), S(2).py(:), S(2).pz(:); ...
    S(1).gate_positions; ...
    S(2).gate_positions  ...
];
margin = 1.0;
xL = [min(all_pos(:,1))-margin, max(all_pos(:,1))+margin];
yL = [min(all_pos(:,2))-margin, max(all_pos(:,2))+margin];
zL = [0.0, max(all_pos(:,3))+margin];

%% -- Figure -------------------------------------------------------------------

figure('Units','centimeters','Position',[2 2 34 14], 'Color','w');
cmap = parula(256);

for k = 1:2
    subplot(1,2,k);
    hold on; grid on; box on;

    data  = S(k);
    px = data.px(:);  py = data.py(:);  pz = data.pz(:);
    v   = sqrt(data.vx(:).^2 + data.vy(:).^2 + data.vz(:).^2);
    gp  = data.gate_positions;
    gn  = data.gate_normals;
    n_g = size(gp, 1);

    % --- trajectory coloured by speed (segment-by-segment) -------------------
    v_idx = max(1, min(256, round(1 + (v / v_all_max) * 255)));
    for i = 1:length(px)-1
        c = cmap(v_idx(i), :);
        plot3(px(i:i+1), py(i:i+1), pz(i:i+1), 'Color', c, 'LineWidth', 2.3);
    end

    % --- gates (circles perpendicular to normal) -----------------------------
    gate_r = double(data.gate_radius);
    th = linspace(0, 2*pi, 60);
    for g = 1:n_g
        p = gp(g,:);  n = gn(g,:);
        n = n / norm(n);
        ref = [0 0 1];
        if abs(dot(n, ref)) > 0.9, ref = [1 0 0]; end
        e1 = cross(n, ref); e1 = e1/norm(e1);
        e2 = cross(n, e1);
        pts = p + gate_r * (cos(th)'*e1 + sin(th)'*e2);

        if g == 1
            col = [1.0, 0.42, 0.21];   % orange - start/finish
        else
            col = [0.18, 0.72, 0.35];  % green
        end
        plot3(pts(:,1), pts(:,2), pts(:,3), 'Color', col, 'LineWidth', 2.0);
        scatter3(p(1), p(2), p(3), 35, [0.75 0.1 0.1], 'filled');
        text(p(1), p(2), p(3)+0.35, sprintf('G%d', g-1), ...
             'FontSize', 8, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

        % tiny arrow showing normal direction
        quiver3(p(1), p(2), p(3), 0.5*n(1), 0.5*n(2), 0.5*n(3), 0, ...
                'Color', [0.4 0.1 0.4], 'LineWidth', 1.0, 'MaxHeadSize', 0.5);
    end

    xlim(xL); ylim(yL); zlim(zL);
    daspect([1 1 1]);
    view(-55, 22);
%    xlabel('X [m]'); ylabe  l('Y [m]'); zlabel('Z [m]');
    set(gca, 'FontSize', 9);
    title(titles{k}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'tex');

    colormap(gca, cmap);
    caxis([0, v_all_max]);
end

% -- Shared colorbar on the right ---------------------------------------------
h = colorbar('Position', [0.935 0.18 0.012 0.70]);
ylabel(h, 'Speed [m/s]', 'FontSize', 10);

% Leave room for the colorbar
set(gcf, 'Position', [2 2 36 14]);

exportgraphics(gcf, '../ACCESS_latex/figs/fig_pmm_circuits.pdf', ...
               'ContentType', 'vector', 'Resolution', 300);
exportgraphics(gcf, 'pmm_circuits_preview.png', 'Resolution', 180);
fprintf('Saved:\n  ../ACCESS_latex/figs/fig_pmm_circuits.pdf\n  pmm_circuits_preview.png\n');
