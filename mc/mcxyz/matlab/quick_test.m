% quick_test.m - Simple test script for MCXYZ with GNU Octave
% This script demonstrates basic loading and visualization of MCXYZ results

clear all; close all; clc;

% Configuration
myname = 'skinvessel';  % Simulation name (change as needed)

fprintf('=== MCXYZ GNU Octave Test ===\n');
fprintf('Testing simulation: %s\n\n', myname);

% Check if required files exist
header_file = sprintf('%s_H.mci', myname);
fluence_file = sprintf('%s_F.bin', myname);

if ~exist(header_file, 'file')
    error('Header file %s not found. Run mcxyz simulation first.', header_file);
end

if ~exist(fluence_file, 'file')
    error('Fluence file %s not found. Run mcxyz simulation first.', fluence_file);
end

%% Load header information
fprintf('Loading header file: %s\n', header_file);
fid = fopen(header_file, 'r');
A = fscanf(fid,'%f',[1 Inf])';
fclose(fid);

% Parse header data
time_min = A(1);
Nx = A(2); Ny = A(3); Nz = A(4);
dx = A(5); dy = A(6); dz = A(7);
mcflag = A(8);
launchflag = A(9);  
boundaryflag = A(10);
xs = A(11); ys = A(12); zs = A(13);

fprintf('Grid dimensions: %dx%dx%d\n', Nx, Ny, Nz);
fprintf('Voxel size: %.4f x %.4f x %.4f cm\n', dx, dy, dz);
fprintf('Simulation time: %.2f minutes\n\n', time_min);

%% Load fluence data
fprintf('Loading fluence data: %s\n', fluence_file);
fid = fopen(fluence_file, 'rb');
if fid == -1
    error('Could not open fluence file: %s', fluence_file);
end

F = fread(fid, 'float32');
fclose(fid);

% Reshape to 3D array
expected_size = Nx * Ny * Nz;
if length(F) ~= expected_size
    error('Fluence data size mismatch. Expected %d, got %d', expected_size, length(F));
end

F = reshape(F, [Ny, Nx, Nz]); % Note: MCXYZ uses [y,x,z] order
fprintf('Loaded fluence array: %dx%dx%d\n', size(F,1), size(F,2), size(F,3));

%% Basic statistics
max_fluence = max(F(:));
mean_fluence = mean(F(:));
nonzero_fluence = sum(F(:) > 0);
total_voxels = numel(F);

fprintf('\n=== Fluence Statistics ===\n');
fprintf('Maximum fluence: %.6f\n', max_fluence);
fprintf('Mean fluence: %.6f\n', mean_fluence);
fprintf('Non-zero voxels: %d / %d (%.1f%%)\n', ...
        nonzero_fluence, total_voxels, 100*nonzero_fluence/total_voxels);

%% Create basic visualizations
fprintf('\nGenerating visualization plots...\n');

% Central slice indices
z_center = round(Nz/2);
y_center = round(Ny/2);
x_center = round(Nx/2);

% Create coordinate arrays (in cm)
x = (1:Nx) * dx;
y = (1:Ny) * dy; 
z = (1:Nz) * dz;

% Figure 1: XY slice (top view)
figure(1); clf;
slice_xy = squeeze(F(:,:,z_center));
imagesc(x, y, slice_xy);
axis image; colorbar;
title(sprintf('%s - XY slice at z=%.3f cm', myname, z(z_center)));
xlabel('x (cm)'); ylabel('y (cm)');

% Figure 2: XZ slice (side view)
figure(2); clf;
slice_xz = squeeze(F(y_center,:,:))';
imagesc(x, z, slice_xz);
axis image; colorbar;
title(sprintf('%s - XZ slice at y=%.3f cm', myname, y(y_center)));
xlabel('x (cm)'); ylabel('z (cm)');

% Figure 3: YZ slice (front view)  
figure(3); clf;
slice_yz = squeeze(F(:,x_center,:))';
imagesc(y, z, slice_yz);
axis image; colorbar;
title(sprintf('%s - YZ slice at x=%.3f cm', myname, x(x_center)));
xlabel('y (cm)'); ylabel('z (cm)');

% Figure 4: 3D visualization (if dataset is small enough)
if Nx <= 100 && Ny <= 100 && Nz <= 100
    fprintf('Creating 3D visualization...\n');
    figure(4); clf;
    
    % Create isosurface at 10% of maximum fluence
    threshold = 0.1 * max_fluence;
    [X,Y,Z] = meshgrid(x, y, z);
    
    if max_fluence > 0
        isosurface(X, Y, Z, F, threshold);
        alpha(0.3);
        xlabel('x (cm)'); ylabel('y (cm)'); zlabel('z (cm)');
        title(sprintf('%s - 3D fluence isosurface (threshold=%.1f%% max)', ...
              myname, 100*threshold/max_fluence));
        axis equal; grid on;
        view(3);
    end
else
    fprintf('Skipping 3D visualization (dataset too large: %dx%dx%d)\n', Nx, Ny, Nz);
end

%% Save results
fprintf('\nSaving analysis results...\n');

% Save plots
figure(1); print('-djpeg', sprintf('%s_test_xy.jpg', myname), '-r150');
figure(2); print('-djpeg', sprintf('%s_test_xz.jpg', myname), '-r150');
figure(3); print('-djpeg', sprintf('%s_test_yz.jpg', myname), '-r150');

% Save data summary
summary_file = sprintf('%s_test_summary.txt', myname);
fid = fopen(summary_file, 'w');
fprintf(fid, 'MCXYZ Test Results Summary\n');
fprintf(fid, '==========================\n\n');
fprintf(fid, 'Simulation: %s\n', myname);
fprintf(fid, 'Grid: %dx%dx%d voxels\n', Nx, Ny, Nz);
fprintf(fid, 'Voxel size: %.4f x %.4f x %.4f cm\n', dx, dy, dz);
fprintf(fid, 'Simulation time: %.2f minutes\n', time_min);
fprintf(fid, '\nFluence Statistics:\n');
fprintf(fid, 'Maximum: %.6f\n', max_fluence);
fprintf(fid, 'Mean: %.6f\n', mean_fluence);
fprintf(fid, 'Non-zero voxels: %d / %d (%.1f%%)\n', ...
        nonzero_fluence, total_voxels, 100*nonzero_fluence/total_voxels);
fclose(fid);

fprintf('Test completed successfully!\n');
fprintf('\nGenerated files:\n');
fprintf('  %s_test_xy.jpg - XY cross-section\n', myname);
fprintf('  %s_test_xz.jpg - XZ cross-section\n', myname);  
fprintf('  %s_test_yz.jpg - YZ cross-section\n', myname);
fprintf('  %s_test_summary.txt - Analysis summary\n', myname);

fprintf('\n=== Test Complete ===\n');
