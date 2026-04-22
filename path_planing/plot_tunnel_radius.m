%% plot_tunnel_radius.m
% Visualización de R(θ) — radio de túnel como función del arc-length
%
% Carga:  path_PMM_results.mat  +  tunnel_radius.mat
% Genera: tunnel_radius_plot.pdf
%
% Uso: correr desde cualquier carpeta (usa rutas absolutas)

clear; close all; clc;

ROOT = '/home/bryansgue/dev/ros2/Tunel-MPCC';

%% ── Cargar datos ─────────────────────────────────────────────────────────────
pmm  = load(fullfile(ROOT, 'path_planing', 'path_PMM_results.mat'));
tun  = load(fullfile(ROOT, 'path_planing', 'tunnel_radius.mat'));

theta_arc    = tun.theta_arc(:);       % [N+1]  longitud de arco en puntos del path
R_theta      = tun.R_theta(:);         % [N+1]  radio en esos puntos
theta_gates  = tun.theta_gates(:);     % [n_gates]
theta_dense  = tun.theta_dense(:);     % [2000]  grid denso
R_dense      = tun.R_dense(:);         % [2000]

R_max       = tun.R_max;
sigma       = tun.sigma;
gate_radius = tun.gate_radius;
n_gates     = round(tun.n_gates);

vel_mag      = pmm.vel_mag(:);         % velocidad a lo largo del path
total_theta  = tun.theta_total;

fprintf('=== Tunnel Radius R(θ) ===\n');
fprintf('  θ_total  = %.2f m\n', total_theta);
fprintf('  R_max    = %.2f m\n', R_max);
fprintf('  R_gate   = %.2f m\n', gate_radius);
fprintf('  sigma    = %.2f m\n', sigma);
fprintf('  n_gates  = %d\n', n_gates);

%% ════════════════════════════════════════════════════════════════════════════
%  FIGURA 1 — R(θ) completo + contexto
%% ════════════════════════════════════════════════════════════════════════════
fig1 = figure('Name','Tunnel Radius R(θ)', 'NumberTitle','off', ...
              'Position',[100 100 1200 800], 'Color','white');

colors_g = lines(n_gates);

%% ── Subplot 1: R(θ) con líneas de gates ─────────────────────────────────────
ax1 = subplot(3,1,1);
hold on; grid on; box on;

% Zona de radio libre (fill)
fill(ax1, [0; theta_dense; total_theta; 0], ...
         [0; R_dense;       0;           0], ...
         [0.85 0.92 1.0], 'EdgeColor','none', 'FaceAlpha', 0.6);

% R(θ) denso
plot(ax1, theta_dense, R_dense, 'b-', 'LineWidth', 2.5, 'DisplayName', 'R(\theta)');

% Línea horizontal gate_radius
yline(ax1, gate_radius, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, ...
      'Label', sprintf('r_{gate}=%.1f m', gate_radius), ...
      'LabelHorizontalAlignment', 'right');

% Línea horizontal R_max
yline(ax1, R_max, ':', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.2, ...
      'Label', sprintf('R_{max}=%.1f m', R_max), ...
      'LabelHorizontalAlignment', 'right');

% Marcadores de gates
for i = 1:n_gates
    xline(ax1, theta_gates(i), '-', 'Color', colors_g(i,:), ...
          'LineWidth', 1.8, 'Alpha', 0.7);
    text(ax1, theta_gates(i), R_max*1.02, sprintf('G%d',i-1), ...
         'FontSize', 8, 'FontWeight', 'bold', 'Color', colors_g(i,:), ...
         'HorizontalAlignment', 'center');
end

ylabel(ax1, 'R(\theta)  [m]', 'FontSize', 11);
title(ax1, sprintf('Tunnel Radius  R(\\theta) — Gaussian CBF  (\\sigma=%.1f m, R_{max}=%.1f m)', ...
      sigma, R_max), 'FontSize', 12);
xlim(ax1, [0 total_theta]);
ylim(ax1, [0 R_max * 1.15]);
legend(ax1, 'Espacio libre', 'R(\theta)', 'Location', 'northeast');

%% ── Subplot 2: Velocidad del path + R(θ) superpuesto ────────────────────────
ax2 = subplot(3,1,2);
yyaxis(ax2, 'left');
hold on; grid on; box on;
plot(ax2, theta_arc, vel_mag, 'k-', 'LineWidth', 1.8, 'DisplayName', '||v(θ)||');
for i = 1:n_gates
    xline(ax2, theta_gates(i), '-', 'Color', colors_g(i,:), ...
          'LineWidth', 1.2, 'Alpha', 0.5);
end
ylabel(ax2, '||v||  [m/s]', 'FontSize', 11);

yyaxis(ax2, 'right');
plot(ax2, theta_dense, R_dense, 'b--', 'LineWidth', 2, 'DisplayName', 'R(\theta)');
ylabel(ax2, 'R(\theta)  [m]', 'FontSize', 11);

xlabel(ax2, '\theta  [m]', 'FontSize', 11);
title(ax2, 'Velocidad PMM y ancho del túnel a lo largo del arc-length', 'FontSize', 11);
xlim(ax2, [0 total_theta]);

%% ── Subplot 3: h(θ) = R²(θ) — CBF barrier value ─────────────────────────────
ax3 = subplot(3,1,3);
hold on; grid on; box on;

h_dense = R_dense.^2;   % valor de la función barrera (sin error de posición)

fill(ax3, [0; theta_dense; total_theta; 0], ...
         [0; h_dense;      0;           0], ...
         [0.9 1.0 0.9], 'EdgeColor','none', 'FaceAlpha', 0.6);

plot(ax3, theta_dense, h_dense, 'g-', 'LineWidth', 2.5);
yline(ax3, gate_radius^2, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, ...
      'Label', sprintf('r^2_{gate}=%.2f', gate_radius^2), ...
      'LabelHorizontalAlignment', 'right');

for i = 1:n_gates
    xline(ax3, theta_gates(i), '-', 'Color', colors_g(i,:), ...
          'LineWidth', 1.5, 'Alpha', 0.7);
end

xlabel(ax3, '\theta  [m]', 'FontSize', 11);
ylabel(ax3, 'h(\theta) = R^2(\theta)  [m^2]', 'FontSize', 11);
title(ax3, 'Función barrera h(\theta) = R^2(\theta)  [drone seguro si ||p_\perp||^2 \leq h(\theta)]', ...
      'FontSize', 11);
xlim(ax3, [0 total_theta]);

sgtitle('Parametrización del Túnel — T-MPCC-CBF', 'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(fig1, fullfile(ROOT,'path_planing','tunnel_radius_plot.pdf'), 'ContentType','vector');
fprintf('Saved: tunnel_radius_plot.pdf\n');

%% ════════════════════════════════════════════════════════════════════════════
%  FIGURA 2 — Vista 3D del túnel sólido con transparencia (surf)
%% ════════════════════════════════════════════════════════════════════════════
fig2 = figure('Name','Tunnel 3D — Solid', 'NumberTitle','off', ...
              'Position',[150 150 1100 750], 'Color','white');
ax4 = axes(fig2);
hold(ax4,'on'); grid(ax4,'on'); box(ax4,'on'); axis(ax4,'equal');
set(ax4, 'Color','white', 'GridColor',[0.7 0.7 0.7], ...
         'XColor','k','YColor','k','ZColor','k');

px = pmm.px(:);
py = pmm.py(:);
pz = pmm.pz(:);
N_pts = length(px);

% ── Construir malla de la superficie del tubo ─────────────────────────────
n_tube = 48;                            % facetas angulares (más = más suave)
phi    = linspace(0, 2*pi, n_tube+1);   % [1 × n_tube+1]

% Tangentes suavizadas (media móvil para evitar quiebres)
dp = [diff(px), diff(py), diff(pz)];
dp = [dp; dp(end,:)];
% Suavizar tangentes con media móvil (sin toolbox)
win = ones(5,1)/5;
for dim = 1:3
    dp(:,dim) = conv(dp(:,dim), win, 'same');
end
tang_norm = dp ./ (sqrt(sum(dp.^2,2)) + 1e-9);  % [N_pts × 3]

% Matrices de superficie: Xs, Ys, Zs  [n_tube+1 × N_pts]
Xs = zeros(n_tube+1, N_pts);
Ys = zeros(n_tube+1, N_pts);
Zs = zeros(n_tube+1, N_pts);
Cs = zeros(n_tube+1, N_pts);   % mapa de color = R(θ) normalizado

for k = 1:N_pts
    tang   = tang_norm(k,:)';
    center = [px(k); py(k); pz(k)];
    R_k    = R_theta(k);

    % Base ortonormal en plano perpendicular a tang
    if abs(tang(3)) < 0.9
        ref = [0;0;1];
    else
        ref = [1;0;0];
    end
    u = cross(tang, ref);  u = u / norm(u);
    v = cross(tang, u);

    % Puntos del anillo a radio R_k
    Xs(:,k) = center(1) + R_k * (cos(phi')*u(1) + sin(phi')*v(1));
    Ys(:,k) = center(2) + R_k * (cos(phi')*u(2) + sin(phi')*v(2));
    Zs(:,k) = center(3) + R_k * (cos(phi')*u(3) + sin(phi')*v(3));

    % Color: 0=estrecho(gate), 1=ancho(libre)
    Cs(:,k) = (R_k - gate_radius) / (R_max - gate_radius + 1e-9);
end

% ── Superficie sólida con transparencia ──────────────────────────────────
hs = surf(ax4, Xs, Ys, Zs, Cs, ...
          'FaceAlpha', 0.30, ...      % transparencia: 0=invisible, 1=sólido
          'EdgeColor', 'none', ...    % sin líneas de malla (limpio)
          'FaceLighting', 'gouraud');
colormap(ax4, cool(256));             % azul(estrecho) → magenta(ancho)
caxis(ax4, [0 1]);
cb2 = colorbar(ax4);
cb2.Color = 'k';
cb2.Label.String = 'R(\theta)  [azul=gate, magenta=libre]';
cb2.Label.Color  = 'k';

% ── Trayectoria central ───────────────────────────────────────────────────
plot3(ax4, px, py, pz, 'k-', 'LineWidth', 2.5);

% ── Gates como discos sólidos semi-transparentes ─────────────────────────
gate_pos = pmm.gate_positions;
gate_nor = pmm.gate_normals;
n_disc = 48;
phi_disc = linspace(0, 2*pi, n_disc+1);

for i = 1:n_gates
    gp = gate_pos(i,:)';
    gn = gate_nor(i,:)';  gn = gn / norm(gn);

    if abs(gn(3)) < 0.9, ref = [0;0;1]; else, ref = [1;0;0]; end
    u = cross(gn, ref);  u = u / norm(u);
    v = cross(gn, u);

    % Disco (fill3 con triángulos desde el centro)
    disc_x = [gp(1)*ones(1,n_disc); ...
               gp(1) + gate_radius*(cos(phi_disc(1:end-1))*u(1) + sin(phi_disc(1:end-1))*v(1)); ...
               gp(1) + gate_radius*(cos(phi_disc(2:end)  )*u(1) + sin(phi_disc(2:end)  )*v(1))];
    disc_y = [gp(2)*ones(1,n_disc); ...
               gp(2) + gate_radius*(cos(phi_disc(1:end-1))*u(2) + sin(phi_disc(1:end-1))*v(2)); ...
               gp(2) + gate_radius*(cos(phi_disc(2:end)  )*u(2) + sin(phi_disc(2:end)  )*v(2))];
    disc_z = [gp(3)*ones(1,n_disc); ...
               gp(3) + gate_radius*(cos(phi_disc(1:end-1))*u(3) + sin(phi_disc(1:end-1))*v(3)); ...
               gp(3) + gate_radius*(cos(phi_disc(2:end)  )*u(3) + sin(phi_disc(2:end)  )*v(3))];

    fill3(ax4, disc_x, disc_y, disc_z, ...
          repmat(colors_g(i,:), n_disc, 1)', ...
          'FaceAlpha', 0.45, 'EdgeColor', 'none');

    % Aro exterior del gate
    [cx,cy,cz] = gate_circle(gp, gn, gate_radius);
    plot3(ax4, cx, cy, cz, '-', 'Color', colors_g(i,:), 'LineWidth', 2.5);

    % Etiqueta
    text(ax4, gp(1), gp(2), gp(3)+0.45, sprintf('G%d',i-1), ...
         'Color','k', 'FontSize',11, 'FontWeight','bold', ...
         'HorizontalAlignment','center');
end

% Inicio / fin
plot3(ax4, px(1),   py(1),   pz(1),   'go', 'MarkerSize',13, ...
      'MarkerFaceColor','g', 'LineWidth',2);
plot3(ax4, px(end), py(end), pz(end), 'rs', 'MarkerSize',13, ...
      'MarkerFaceColor','r', 'LineWidth',2);

% Iluminación para dar sensación de volumen
light(ax4, 'Position',[1 0.5 1]*20, 'Style','infinite');
lighting(ax4, 'gouraud');
material(ax4, 'dull');

xlabel(ax4,'X [m]','FontSize',12,'Color','k');
ylabel(ax4,'Y [m]','FontSize',12,'Color','k');
zlabel(ax4,'Z [m]','FontSize',12,'Color','k');
title(ax4, sprintf('Tunnel T-MPCC-CBF  |  R_{max}=%.1f m  r_{gate}=%.1f m  \\sigma=%.1f m', ...
      R_max, gate_radius, sigma), ...
      'FontSize',13,'FontWeight','bold','Color','k');
view(ax4, 40, 22);

exportgraphics(fig2, fullfile(ROOT,'path_planing','tunnel_3D.pdf'), 'ContentType','vector');
fprintf('Saved: tunnel_3D.pdf\n');
fprintf('\nDone.\n');

%% ════════════════════════════════════════════════════════════════════════════
%  LOCAL FUNCTIONS
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
