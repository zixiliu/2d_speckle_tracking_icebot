clearvars;

c0 = 1540;                      % [m/s]
% set the size of the perfectly matched layer (PML)
pml_x_size = 15;                % [grid points]
pml_y_size = 10;                % [grid points]
pml_z_size = 10;                % [grid points]

% set total number of grid points not including the PML
sc = 1;
Nx = 128/sc - 2*pml_x_size;     % [grid points]
Ny = 128/sc - 2*pml_y_size;     % [grid points]
Nz = 64/sc - 2*pml_z_size;     % [grid points]

% set desired grid size in the x-direction not including the PML
x = 50e-3;                      % [m]

% calculate the spacing between the grid points
dx = x / Nx;                    % [m]
dy = dx;                        % [m]
dz = dx;  


scattering_map = randn([Nx, Ny, Nz]);

scattering_c0 = c0 + 25 + 75 * scattering_map;
scattering_c0(scattering_c0 > 1600) = 1600;
scattering_c0(scattering_c0 < 1400) = 1400;

radius = 6e-3;
x_pos = 32e-3; 
y_pos = dy * Ny/2;
scattering_region1 = makeBall(Nx, Ny, Nz, round(x_pos / dx), round(y_pos / dx), Nz/2, round(radius / dx));

the_ball = scattering_c0(scattering_region1 == 1);


y_offset = 5;
sim(y_offset,'example_us_phased_array_scan_lines_yp5', 'ryp5_full.fig', 'ryp5.fig', the_ball);

y_offset = 4;
sim(y_offset,'example_us_phased_array_scan_lines_yp4', 'ryp4_full.fig', 'ryp4.fig', the_ball);

y_offset = 3;
sim(y_offset,'example_us_phased_array_scan_lines_yp3', 'ryp3_full.fig', 'ryp3.fig', the_ball);

y_offset = 2;
sim(y_offset,'example_us_phased_array_scan_lines_yp2', 'ryp2_full.fig', 'ryp2.fig', the_ball);

y_offset = 1;
sim(y_offset,'example_us_phased_array_scan_lines_yp1', 'ryp1_full.fig', 'ryp1.fig', the_ball);

y_offset = 0;
sim(y_offset,'example_us_phased_array_scan_lines_0', 'r0_full.fig', 'r0.fig', the_ball);


y_offset = -5;
sim(y_offset,'example_us_phased_array_scan_lines_ym5', 'rym5_full.fig', 'rym5 .fig', the_ball);

y_offset = -4;
sim(y_offset,'example_us_phased_array_scan_lines_ym4', 'rym4_full.fig', 'rym4.fig', the_ball);

y_offset = -3;
sim(y_offset,'example_us_phased_array_scan_lines_ym3', 'rym3_full.fig', 'rym3.fig', the_ball);

y_offset = -2;
sim(y_offset,'example_us_phased_array_scan_lines_ym2', 'rym2_full.fig', 'rym2.fig', the_ball);

y_offset = -1;
sim(y_offset,'example_us_phased_array_scan_lines_ym1', 'rym1_full.fig', 'rym1.fig', the_ball);


