# Testing MCXYZ with GNU Octave

This guide shows how to test and visualize MCXYZ Monte Carlo simulations using GNU Octave (free alternative to MATLAB).

## Prerequisites

1. **Install GNU Octave**: Download from https://www.gnu.org/software/octave/
2. **Built MCXYZ**: Ensure you have `mcxyz.exe` in the `bin/` directory

## Quick Test with Existing Data

The easiest way to start is with the existing `skinvessel` example data:

### Step 1: Run MCXYZ Simulation

```bash
cd "d:\github-wip\MCML\mc\mcxyz\res"
..\bin\mcxyz.exe --threads 8 --verbose skinvessel
```

This will use the existing:

- `skinvessel_H.mci` (header file with simulation parameters)
- `skinvessel_T.bin` (tissue structure file)

And generate:

- `skinvessel_F.bin` (fluence distribution)
- `skinvessel_props.m` (optical properties)

### Step 2: Launch GNU Octave

```bash
# Start Octave in the matlab directory
cd "d:\github-wip\MCML\mc\mcxyz\matlab"
octave
```

### Step 3: Visualize Results

In Octave, run the visualization script:

```octave
% Add the res directory to path so Octave can find the data files
addpath('../res');

% Load and visualize the results
lookmcxyz
```

This will create several visualization files:

- `skinvessel_tissue.jpg` - Shows tissue structure
- `skinvessel_Fzx.jpg` - Fluence vs z,x cross-section
- `skinvessel_Fzy.jpg` - Fluence vs z,y cross-section

## Creating Your Own Test Case

### Step 1: Create Custom Tissue Geometry

In Octave:

```octave
% Add paths
addpath('../res');
cd('../res');  % Change to res directory

% Edit parameters in maketissue.m (or create a copy)
edit('../matlab/maketissue.m')
```

Key parameters to modify in `maketissue.m`:

- `myname = 'mytest';` - Your simulation name
- `time_min = 2;` - Simulation time in minutes  
- `Nbins = 100;` - Grid size (100x100x100 voxels)
- `binsize = 0.001;` - Voxel size in cm
- `nm = 630;` - Wavelength in nm

### Step 2: Generate Input Files

```octave
% Run the tissue generation script
cd('../matlab');
addpath('../res');
maketissue
```

This creates:

- `mytest_H.mci` - Header file
- `mytest_T.bin` - Tissue structure file

### Step 3: Run Simulation

```bash
cd "d:\github-wip\MCML\mc\mcxyz\res"
..\bin\mcxyz.exe --threads 8 --verbose mytest
```

### Step 4: Analyze Results

```octave
% In Octave, modify the script to use your data
cd('../matlab');
addpath('../res');

% Edit lookmcxyz.m to change myname
edit('lookmcxyz.m')
% Change line: myname = 'mytest';

% Run visualization
lookmcxyz
```

## Advanced Testing Workflows

### Performance Comparison Test

Test different execution modes:

```bash
# Single-threaded (baseline)
time ..\bin\mcxyz.exe skinvessel

# Multi-threaded 
time ..\bin\mcxyz.exe --threads 8 skinvessel

# Ultra-optimized
time ..\bin\mcxyz.exe --ultra --threads 0 skinvessel
```

### Reproducibility Test

```bash
# Run with specific seed for reproducible results
..\bin\mcxyz.exe --seed 12345 --photons 100000 skinvessel
```

### Parameter Sweep

Create multiple test cases with different optical properties:

```octave
% In maketissue.m, create a loop
for mua = [0.1, 0.5, 1.0]
    myname = sprintf('test_mua_%g', mua);
    % Modify tissue properties...
    % Run maketissue
end
```

## Useful Octave Commands

### Loading and Examining Data

```octave
% Load fluence data
filename = 'skinvessel_F.bin';
fid = fopen(filename, 'rb');
F = fread(fid, 'float32');
fclose(fid);

% Reshape to 3D array (adjust dimensions as needed)
F = reshape(F, [200, 200, 200]);

% Basic statistics
max_fluence = max(F(:))
mean_fluence = mean(F(:))
total_energy = sum(F(:))
```

### Custom Visualization

```octave
% Create custom cross-section plots
figure;
imagesc(squeeze(F(:,:,100))); % XY slice at z=100
colorbar;
title('Fluence Distribution XY slice');

figure;
imagesc(squeeze(F(:,100,:))); % XZ slice at y=100  
colorbar;
title('Fluence Distribution XZ slice');
```

### Data Export

```octave
% Export to CSV for external analysis
csvwrite('fluence_data.csv', F(:,:,100));

% Save workspace
save('mcxyz_results.mat');
```

## Troubleshooting

### Common Issues

1. **"File not found" errors**: Make sure you're in the correct directory and paths are set
2. **Memory errors**: Reduce grid size (Nbins) for large simulations
3. **Slow performance**: Use multi-threading options (`--threads 8`)

### Octave vs MATLAB Differences

Most scripts work identically, but note:

- Use `addpath()` instead of `path()` for adding directories
- Some plot formatting functions may differ slightly
- Use `pkg install -forge package_name` to install additional packages if needed

## Example Complete Workflow

```bash
# 1. Build the project
cd "d:\github-wip\MCML\mc\mcxyz\build"
make clean && make

# 2. Start Octave
cd ../matlab
octave

# 3. In Octave:
addpath('../res');
cd('../res');

# 4. Run simulation
system('../bin/mcxyz.exe --threads 8 --verbose skinvessel');

# 5. Visualize
cd('../matlab');
lookmcxyz;

# 6. Check generated images
system('ls -la ../res/skinvessel_*.jpg');
```

This workflow will give you a complete test of the MCXYZ system with visual results showing the tissue structure and light propagation patterns.
