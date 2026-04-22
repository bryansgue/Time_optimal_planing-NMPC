% animate_flatness_ref.m  -  Animate differential-flatness attitude reference.
%
% Plays back a quadrotor rendered with its body axes (x_b red, y_b green,
% z_b blue) at every sample of the flatness reference. Shows the reference
% trajectory traced so far, current thrust, bank angle, and time.
%
% Usage:
%   Set CIRCUIT = 'fig8' or 'loop' below and run.
%   Or call animate_flatness_ref('loop') after wrapping into a function.
%
% Inputs expected in path_planing/:
%   flatness_ref_fig8.mat
%   flatness_ref_loop.mat
%
% Run from:  path_planing/

close all; clc;

% ------- CONFIG --------------------------------------------------------------
CIRCUIT      = 'fig8';    % 'fig8' or 'loop'
PLAYBACK     = 0.25;       % 1.0 = real time, 0.5 = half speed, 2.0 = double
LOOP_FOREVER = true;      % repeat animation in a loop (ignored if SAVE_GIF=true)
ARM_LEN      = 0.25;      % quadrotor half-arm length [m] (visual only)
AXIS_LEN     = 0.45;      % body-axes arrow length [m]

% ------- GIF EXPORT ---------------------------------------------------------
SAVE_GIF     = true;       % if true, write an animated GIF and exit after one pass
GIF_FILE     = sprintf('animate_flatness_ref_%s.gif', CIRCUIT);
GIF_STEP     = 2;          % keep 1 of every GIF_STEP frames (speed/size tradeoff)
GIF_FPS      = 25;         % playback fps encoded in the GIF

% ------- LOAD REFERENCE ------------------------------------------------------
fname = sprintf('flatness_ref_%s.mat', CIRCUIT);
if ~isfile(fname)
    error('Run export_flatness_mat.py first. Missing: %s', fname);
end
D = load(fname);

t  = D.t(:);        N = length(t);
p  = D.p;           % 3xN
v  = D.v;           % 3xN
q  = D.q;           % 4xN (qw,qx,qy,qz)
T  = D.T(:);        % Nx1
om = D.omega;       % 3xN
xb = D.xb; yb = D.yb; zb = D.zb;   % 3xN each
psi = D.yaw(:);

% ------- WORKSPACE BOX FOR CAMERA ------------------------------------------
m = 0.8;
xL = [min(p(1,:))-m, max(p(1,:))+m];
yL = [min(p(2,:))-m, max(p(2,:))+m];
zL = [max(0, min(p(3,:))-m),  max(p(3,:))+m];

% ------- FIGURE SETUP --------------------------------------------------------
fig = figure('Units','centimeters','Position',[2 2 30 18], 'Color','w', ...
             'Name',sprintf('Flatness reference - %s', CIRCUIT));

% 3D animation panel
ax3d = subplot('Position',[0.05 0.30 0.55 0.65]);
hold(ax3d,'on'); grid(ax3d,'on'); box(ax3d,'on');
axis(ax3d,'equal');
xlim(ax3d, xL); ylim(ax3d, yL); zlim(ax3d, zL);
view(ax3d, -55, 22);
xlabel(ax3d,'X [m]'); ylabel(ax3d,'Y [m]'); zlabel(ax3d,'Z [m]');
title(ax3d, sprintf('Flatness attitude reference -- %s', upper(CIRCUIT)), ...
      'FontWeight','bold');

% Enable mouse rotation on the 3D axis only (side plots stay 2D)
rotate3d(ax3d, 'on');
% Keep axes ratio stable while rotating
set(ax3d, 'DataAspectRatioMode','manual', 'PlotBoxAspectRatioMode','manual');

% Reference path in grey (full trajectory as a guide)
plot3(ax3d, p(1,:), p(2,:), p(3,:), ':', 'Color',[0.6 0.6 0.6], 'LineWidth',1.0);

% Dynamic objects (created empty, updated each frame)
h_trace = plot3(ax3d, nan, nan, nan, '-', 'Color',[0.1 0.4 0.9], 'LineWidth',2.0);
h_xb = quiver3(ax3d, 0,0,0, 0,0,0, 0, 'Color','r', 'LineWidth',2.0, 'MaxHeadSize',0.5);
h_yb = quiver3(ax3d, 0,0,0, 0,0,0, 0, 'Color',[0 0.6 0], 'LineWidth',2.0, 'MaxHeadSize',0.5);
h_zb = quiver3(ax3d, 0,0,0, 0,0,0, 0, 'Color','b', 'LineWidth',2.5, 'MaxHeadSize',0.5);
h_tht= quiver3(ax3d, 0,0,0, 0,0,0, 0, 'Color',[0.9 0.4 0], 'LineWidth',2.0, ...
               'MaxHeadSize',0.5, 'LineStyle','--');   % thrust vector (same as zb, scaled)

% --- side panels: time-history plots (static, full trace; moving cursor) -----
% Thrust
axT = subplot('Position',[0.66 0.70 0.32 0.25]);
plot(axT, t, T, 'Color',[0.2 0.2 0.2]); grid(axT,'on');
ylabel(axT,'T [N]'); title(axT,'Thrust reference');
hold(axT,'on'); cT = plot(axT, nan, nan, 'ro','MarkerFaceColor','r');

% Quaternion components
axQ = subplot('Position',[0.66 0.40 0.32 0.25]);
plot(axQ, t, q(1,:), 'k-', 'DisplayName','q_w'); hold(axQ,'on');
plot(axQ, t, q(2,:), 'r-', 'DisplayName','q_x');
plot(axQ, t, q(3,:), 'g-', 'DisplayName','q_y');
plot(axQ, t, q(4,:), 'b-', 'DisplayName','q_z');
grid(axQ,'on'); legend(axQ,'Location','southoutside','Orientation','horizontal');
ylabel(axQ,'q'); title(axQ,'Attitude quaternion reference');
cQ = plot(axQ, nan, nan, 'ko','MarkerFaceColor','k');

% Body rates
axW = subplot('Position',[0.66 0.10 0.32 0.22]);
plot(axW, t, om(1,:), 'r-', 'DisplayName','\omega_x'); hold(axW,'on');
plot(axW, t, om(2,:), 'g-', 'DisplayName','\omega_y');
plot(axW, t, om(3,:), 'b-', 'DisplayName','\omega_z');
grid(axW,'on'); legend(axW,'Location','southoutside','Orientation','horizontal');
ylabel(axW,'\omega [rad/s]'); xlabel(axW,'t [s]');
title(axW,'Body angular velocity reference');
cW = plot(axW, nan, nan, 'ko','MarkerFaceColor','k');

% Text box with live metrics
hTxt = annotation('textbox',[0.05 0.04 0.55 0.18],'EdgeColor',[0.85 0.85 0.85], ...
                  'BackgroundColor',[0.98 0.98 0.98],'FontName','Courier', ...
                  'FontSize',10,'String','');

% Status bar
hTime = annotation('textbox',[0.05 0.97 0.55 0.025],'EdgeColor','none', ...
                   'FontWeight','bold','FontSize',11,'String','', ...
                   'HorizontalAlignment','center');

% ------- ANIMATION LOOP ------------------------------------------------------
T_total = t(end);
dt_samp = mean(diff(t));
frame_pause = dt_samp / PLAYBACK;

if SAVE_GIF
    LOOP_FOREVER = false;
    gif_delay = 1.0 / GIF_FPS;
    gif_first = true;
    fprintf('Recording GIF to %s (step=%d, fps=%d)...\n', GIF_FILE, GIF_STEP, GIF_FPS);
end

do_loop = true;
while do_loop
    for k = 1:N
        if ~isvalid(fig); return; end

        pk  = p(:,k);
        xbk = xb(:,k); ybk = yb(:,k); zbk = zb(:,k);

        % Traced path up to k
        set(h_trace, 'XData',p(1,1:k), 'YData',p(2,1:k), 'ZData',p(3,1:k));

        % Body axes arrows
        set(h_xb, 'XData',pk(1),'YData',pk(2),'ZData',pk(3), ...
                  'UData',AXIS_LEN*xbk(1),'VData',AXIS_LEN*xbk(2),'WData',AXIS_LEN*xbk(3));
        set(h_yb, 'XData',pk(1),'YData',pk(2),'ZData',pk(3), ...
                  'UData',AXIS_LEN*ybk(1),'VData',AXIS_LEN*ybk(2),'WData',AXIS_LEN*ybk(3));
        set(h_zb, 'XData',pk(1),'YData',pk(2),'ZData',pk(3), ...
                  'UData',AXIS_LEN*zbk(1),'VData',AXIS_LEN*zbk(2),'WData',AXIS_LEN*zbk(3));

        % Thrust vector: thrust is along +z_b with magnitude T
        tht_scale = 0.08;   % m per N for visualization
        set(h_tht,'XData',pk(1),'YData',pk(2),'ZData',pk(3), ...
                  'UData',tht_scale*T(k)*zbk(1), ...
                  'VData',tht_scale*T(k)*zbk(2), ...
                  'WData',tht_scale*T(k)*zbk(3));

        % Cursor markers on side plots
        set(cT,'XData',t(k),'YData',T(k));
        set(cQ,'XData',[t(k) t(k) t(k) t(k)],'YData',q(:,k)');
        set(cW,'XData',[t(k) t(k) t(k)],'YData',om(:,k)');

        % Metrics
        bank_deg  = acosd(max(-1,min(1,zbk(3))));   % tilt from world-up
        inv_flag  = '';
        if zbk(3) < 0
            inv_flag = '  <-- INVERTED!';
        end
        msg = sprintf([ ...
            ' t = %5.2f / %.2f s   k = %3d/%d\n'    ...
            ' p      = [%+5.2f %+5.2f %+5.2f] m\n'  ...
            ' v      = [%+5.2f %+5.2f %+5.2f] m/s  |v|=%.2f m/s\n' ...
            ' q      = [%+5.2f %+5.2f %+5.2f %+5.2f]\n' ...
            ' bank   = %5.1f deg%s\n'  ...
            ' T      = %5.2f N        (hover = 10.6 N)\n' ...
            ' omega  = [%+5.2f %+5.2f %+5.2f] rad/s'], ...
            t(k), T_total, k, N, ...
            pk(1),pk(2),pk(3), ...
            v(1,k),v(2,k),v(3,k), norm(v(:,k)), ...
            q(1,k),q(2,k),q(3,k),q(4,k), ...
            bank_deg, inv_flag, T(k), om(1,k),om(2,k),om(3,k));
        set(hTxt,'String',msg);

        set(hTime,'String', sprintf('Playback %.1fx   |   frame %d/%d', ...
                                    PLAYBACK, k, N));
        drawnow;    % full drawnow so rotate3d mouse events are processed

        if SAVE_GIF
            if mod(k-1, GIF_STEP) == 0 || k == N
                frame = getframe(fig);
                [A, map] = rgb2ind(frame.cdata, 256);
                if gif_first
                    imwrite(A, map, GIF_FILE, 'gif', ...
                            'LoopCount', Inf, 'DelayTime', gif_delay);
                    gif_first = false;
                else
                    imwrite(A, map, GIF_FILE, 'gif', ...
                            'WriteMode', 'append', 'DelayTime', gif_delay);
                end
            end
        else
            pause(frame_pause);
        end
    end
    do_loop = LOOP_FOREVER && isvalid(fig);
end

if SAVE_GIF
    fprintf('GIF saved: %s\n', GIF_FILE);
end
