%% plot_path_PMM.m
% Visualización del path planning PMM (Point Mass Model)
% Carga path_PMM_results.mat y genera figuras 3D + perfiles
%
% Uso:
%   cd <ruta>/path_planing
%   plot_path_PMM
%
% Generado automáticamente — corresponde a path_time_3D_PMM.py

clear; close all; clc;

%% ── Cargar datos ─────────────────────────────────────────────────────────────
ROOT = '/home/bryansgue/dev/ros2/Tunel-MPCC';
data = load(fullfile(ROOT, 'path_planing', 'path_PMM_results.mat'));

px  = data.px(:);
py  = data.py(:);
pz  = data.pz(:);
vx  = data.vx(:);
vy  = data.vy(:);
vz  = data.vz(:);
ax  = data.ax(:);
ay  = data.ay(:);
az  = data.az(:);

vel_mag = data.vel_mag(:);
acc_mag = data.acc_mag(:);
t       = data.t_traj(:);
t_ctrl  = t(1:end-1);           % vector tiempo para controles [N]

gate_pos    = data.gate_positions;   % [n_gates × 3]
gate_nor    = data.gate_normals;     % [n_gates × 3]
gate_r      = data.gate_radius;
safety_r    = gate_r - data.safety_margin;
n_gates     = size(gate_pos, 1);
T_opt       = data.T_opt;

fprintf('=== PMM Path Planning Results ===\n');
fprintf('  Tiempo óptimo : %.2f s\n', T_opt);
fprintf('  Distancia total: %.1f m\n', sum(sqrt(diff(px).^2 + diff(py).^2 + diff(pz).^2)));
fprintf('  Vel máxima     : %.2f m/s\n', max(vel_mag));
fprintf('  Acc máxima     : %.2f m/s²\n', max(acc_mag));
fprintf('  Gates          : %d\n', n_gates);

%% ════════════════════════════════════════════════════════════════════════════
%  FIGURA 1 — Panel 2×2
%% ════════════════════════════════════════════════════════════════════════════
fig1 = figure('Name','PMM Path Planning — Panel', 'NumberTitle','off', ...
              'Position',[100 100 1400 900], 'Color','white');

%% ── Subplot 1: Trayectoria 3D ────────────────────────────────────────────────
ax1 = subplot(2,2,1);
hold on; grid on; box on; axis equal;

% Trayectoria coloreada por velocidad
npts = length(px);
cmap = colormap(ax1, parula(256));
v_norm = (vel_mag - min(vel_mag)) / (max(vel_mag) - min(vel_mag) + 1e-9);
for k = 1:npts-1
    ci = max(1, round(v_norm(k)*255) + 1);
    plot3(ax1, px(k:k+1), py(k:k+1), pz(k:k+1), ...
          'Color', cmap(ci,:), 'LineWidth', 2);
end

% Gates
colors_g = lines(n_gates);
for i = 1:n_gates
    [cx,cy,cz] = gate_circle(gate_pos(i,:)', gate_nor(i,:)', gate_r);
    [sx,sy,sz] = gate_circle(gate_pos(i,:)', gate_nor(i,:)', safety_r);
    plot3(ax1, cx, cy, cz, 'Color', colors_g(i,:), 'LineWidth', 2.5);
    plot3(ax1, sx, sy, sz, '--', 'Color', colors_g(i,:), 'LineWidth', 1);
    text(gate_pos(i,1), gate_pos(i,2), gate_pos(i,3)+0.3, ...
         sprintf('G%d',i-1), 'FontSize',9,'FontWeight','bold', ...
         'Color', colors_g(i,:), 'HorizontalAlignment','center', 'Parent', ax1);
    % Normal arrow
    quiver3(ax1, gate_pos(i,1), gate_pos(i,2), gate_pos(i,3), ...
            gate_nor(i,1)*0.6, gate_nor(i,2)*0.6, gate_nor(i,3)*0.6, ...
            'Color', colors_g(i,:), 'LineWidth', 1.5, 'MaxHeadSize', 0.8);
end

% Start / End markers
plot3(ax1, px(1),   py(1),   pz(1),   'go', 'MarkerSize',10, 'MarkerFaceColor','g');
plot3(ax1, px(end), py(end), pz(end), 'rs', 'MarkerSize',10, 'MarkerFaceColor','r');

cb = colorbar(ax1);
cb.Label.String = 'Speed [m/s]';
caxis(ax1, [min(vel_mag), max(vel_mag)]);
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('3D Trajectory (coloured by speed)');
view(45, 25);

%% ── Subplot 2: Perfil de velocidad ──────────────────────────────────────────
ax2 = subplot(2,2,2);
hold on; grid on; box on;
plot(ax2, t, vel_mag, 'b-', 'LineWidth', 2);
% Marcar instantes de gate
k_gate = round(data.k_gate) + 1;   % 1-indexed
for i = 1:n_gates
    ki = min(k_gate(i), length(t));
    xline(ax2, t(ki), '--', 'Color', colors_g(i,:), 'LineWidth', 1.2, ...
          'Label', sprintf('G%d',i-1), 'LabelOrientation','horizontal', ...
          'FontSize', 8);
end
xlabel('Time [s]'); ylabel('||v|| [m/s]');
title('Speed profile');

%% ── Subplot 3: Perfil de aceleración ────────────────────────────────────────
ax3 = subplot(2,2,3);
hold on; grid on; box on;
plot(ax3, t_ctrl, ax, 'r-',  'LineWidth', 1.5, 'DisplayName', 'ax');
plot(ax3, t_ctrl, ay, 'g-',  'LineWidth', 1.5, 'DisplayName', 'ay');
plot(ax3, t_ctrl, az, 'b-',  'LineWidth', 1.5, 'DisplayName', 'az');
plot(ax3, t_ctrl, acc_mag, 'k--', 'LineWidth', 2, 'DisplayName', '||a||');
legend('Location','northeast');
xlabel('Time [s]'); ylabel('Acceleration [m/s²]');
title('Control inputs (accelerations)');

%% ── Subplot 4: Jerk ─────────────────────────────────────────────────────────
ax4 = subplot(2,2,4);
hold on; grid on; box on;
dt_opt = data.dt_opt;
jx = diff(ax) / dt_opt;
jy = diff(ay) / dt_opt;
jz = diff(az) / dt_opt;
jmag = sqrt(jx.^2 + jy.^2 + jz.^2);
t_jerk = t_ctrl(1:end-1);
plot(ax4, t_jerk, jmag, 'm-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Jerk [m/s³]');
title(sprintf('Jerk magnitude  (max=%.2f m/s³)', max(jmag)));

sgtitle(sprintf('PMM Path Planning  |  T_{opt}=%.2fs  |  v_{max}=%.2f m/s', ...
        T_opt, max(vel_mag)), 'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(fig1, fullfile(ROOT,'path_planing','path_PMM_panel.pdf'), 'ContentType','vector');
fprintf('Saved: path_PMM_panel.pdf\n');

%% ════════════════════════════════════════════════════════════════════════════
%  FIGURA 2 — 3D standalone alta calidad
%% ════════════════════════════════════════════════════════════════════════════
fig2 = figure('Name','PMM — 3D Detail', 'NumberTitle','off', ...
              'Position',[150 150 900 700], 'Color','white');
ax5 = axes(fig2);
hold(ax5,'on'); grid(ax5,'on'); box(ax5,'on'); axis(ax5,'equal');

% Trayectoria coloreada por velocidad
cmap2 = plasma_cmap(256);
for k = 1:npts-1
    ci = max(1, round(v_norm(k)*255) + 1);
    plot3(ax5, px(k:k+1), py(k:k+1), pz(k:k+1), ...
          'Color', cmap2(ci,:), 'LineWidth', 3);
end

% Gates
for i = 1:n_gates
    [cx,cy,cz] = gate_circle(gate_pos(i,:)', gate_nor(i,:)', gate_r);
    plot3(ax5, cx, cy, cz, 'w-',  'LineWidth', 3);
    plot3(ax5, cx, cy, cz, '--',  'Color', [0.8 0.8 0.8], 'LineWidth', 1);
    text(gate_pos(i,1), gate_pos(i,2), gate_pos(i,3)+0.4, ...
         sprintf('G%d',i-1), 'Color','white', 'FontSize',11, ...
         'FontWeight','bold', 'HorizontalAlignment','center', 'Parent', ax5);
end

plot3(ax5, px(1),   py(1),   pz(1),   'go', 'MarkerSize',12, 'MarkerFaceColor','g', 'LineWidth',2);
plot3(ax5, px(end), py(end), pz(end), 'rs', 'MarkerSize',12, 'MarkerFaceColor','r', 'LineWidth',2);

cb2 = colorbar(ax5);
cb2.Label.String = 'Speed [m/s]';
cb2.Color = 'k';
caxis(ax5, [min(vel_mag), max(vel_mag)]);

xlabel(ax5,'X [m]','FontSize',13);
ylabel(ax5,'Y [m]','FontSize',13);
zlabel(ax5,'Z [m]','FontSize',13);
title(ax5, sprintf('PMM Minimum-Time Path  —  %d gates  |  T_{opt}=%.2f s', ...
      n_gates, T_opt), 'FontSize', 14, 'FontWeight','bold');
view(ax5, 45, 25);

exportgraphics(fig2, fullfile(ROOT,'path_planing','path_PMM_3D_detail.pdf'), 'ContentType','vector');
fprintf('Saved: path_PMM_3D_detail.pdf\n');

%% ════════════════════════════════════════════════════════════════════════════
%  FIGURA 3 — Proyecciones XY / XZ / YZ
%% ════════════════════════════════════════════════════════════════════════════
fig3 = figure('Name','PMM — Projections', 'NumberTitle','off', ...
              'Position',[200 200 1200 400], 'Color','white');

proj_labels = {'XY','XZ','YZ'};
x_data = {px, px, py};
y_data = {py, pz, pz};
xlab   = {'X [m]','X [m]','Y [m]'};
ylab   = {'Y [m]','Z [m]','Z [m]'};

for s = 1:3
    axs = subplot(1,3,s);
    hold on; grid on; box on; axis equal;
    xd = x_data{s};  yd = y_data{s};
    for k = 1:length(xd)-1
        ci = max(1, round(v_norm(k)*255) + 1);
        plot(axs, xd(k:k+1), yd(k:k+1), 'Color', cmap(ci,:), 'LineWidth', 2);
    end
    for i = 1:n_gates
        plot(axs, gate_pos(i, strcmp(proj_labels{s}(1),'X')*1 + strcmp(proj_labels{s}(1),'Y')*2), ...
                  gate_pos(i, strcmp(proj_labels{s}(2),'Y')*2 + strcmp(proj_labels{s}(2),'Z')*3), ...
             'k+', 'MarkerSize',12, 'LineWidth',2);
    end
    xlabel(xlab{s}); ylabel(ylab{s});
    title(sprintf('Projection %s', proj_labels{s}));
end
sgtitle('Path projections (coloured by speed)', 'FontSize',13);

exportgraphics(fig3, fullfile(ROOT,'path_planing','path_PMM_projections.pdf'), 'ContentType','vector');
fprintf('Saved: path_PMM_projections.pdf\n');

fprintf('\nDone. Open path_PMM_panel.pdf, path_PMM_3D_detail.pdf, path_PMM_projections.pdf\n');

%% ════════════════════════════════════════════════════════════════════════════
%  LOCAL FUNCTIONS  (must be at end of script)
%% ════════════════════════════════════════════════════════════════════════════

function [cx, cy, cz] = gate_circle(center, normal, radius, n)
    if nargin < 4, n = 64; end
    normal = normal(:) / norm(normal);
    if abs(normal(1)) < 0.9
        u = cross(normal, [1;0;0]);
    else
        u = cross(normal, [0;1;0]);
    end
    u = u / norm(u);
    v = cross(normal, u);
    theta = linspace(0, 2*pi, n);
    pts = center(:) + radius * (cos(theta) .* u + sin(theta) .* v);
    cx = pts(1,:);
    cy = pts(2,:);
    cz = pts(3,:);
end

function cmap = plasma_cmap(n)
    if nargin < 1, n = 256; end
    try
        cmap = colormap(plasma(n));
    catch
        t = linspace(0,1,n)';
        r = max(0, min(1,  0.050 + 2.380*t - 1.580*t.^2));
        g = max(0, min(1,  0.030 - 0.600*t + 1.800*t.^2));
        b = max(0, min(1,  0.530 + 0.400*t - 1.700*t.^2));
        cmap = max(0, min(1, [r, g, b]));
    end
end
