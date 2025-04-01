#!/usr/bin/env python3
"""
SPECFEM3D simulation runner.

This module provides utilities to set up and run SPECFEM3D simulations,
including preparing Par_file, running the mesher, and solver.
"""
import os
import subprocess
import logging
from pathlib import Path
import shutil
import re
import numpy as np

logger = logging.getLogger(__name__)

class SpecfemSimulation:
    """Class to handle SPECFEM3D simulations."""
    
    def __init__(self, specfem_dir, output_dir, nproc=32, use_gpu=False):
        """
        Initialize SPECFEM3D simulation handler.
        
        Args:
            specfem_dir (str): Path to SPECFEM3D installation
            output_dir (str): Path to store simulation outputs
            nproc (int): Number of MPI processes to use
            use_gpu (bool): Whether to use GPU acceleration
        """
        # Handle relative and user paths
        specfem_dir = os.path.abspath(os.path.expanduser(specfem_dir))
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        
        # Verify SPECFEM3D installation
        if not os.path.exists(os.path.join(specfem_dir, "bin", "xmeshfem3D")):
            logger.warning(f"Warning: SPECFEM3D executables not found at {specfem_dir}/bin/. "
                          "Make sure SPECFEM3D is properly installed and compiled.")
        
        self.specfem_dir = Path(specfem_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set the number of MPI processes
        self.nproc = nproc
        self.use_gpu = use_gpu
        
    def generate_mesh_par_file(self, domain_size=(4.0, 4.0, 4.0), layers=None, output_path=None):
        """Generate Mesh_Par_file for SPECFEM3D.
        
        Args:
            domain_size (tuple): Domain size in km (x, y, z)
            layers (list): List of layer configurations from bottom to top
                           Each layer is a dict with keys: thickness, rho, vp, vs, Qkappa, Qmu
            output_path (str): Path to save the Mesh_Par_file
                              
        Returns:
            str: Path to the generated Mesh_Par_file
        """
        # Use default layers if none provided
        if layers is None:
            layers = [
                {"thickness": 1000.0, "rho": 2500.0, "vp": 3500.0, "vs": 2000.0, "Qkappa": 9999.0, "Qmu": 9999.0},  # bottom 1 km
                {"thickness": 2000.0, "rho": 2200.0, "vp": 2500.0, "vs": 1200.0, "Qkappa": 9999.0, "Qmu": 9999.0},  # middle 2 km
                {"thickness": 1000.0, "rho": 2000.0, "vp": 1500.0, "vs":  500.0, "Qkappa": 9999.0, "Qmu": 9999.0}   # top 1 km
            ]
        
        total_depth = sum(layer["thickness"] for layer in layers)
        assert abs(total_depth - domain_size[2]*1000.0) < 1e-6, f"Layer thicknesses must sum to {domain_size[2]} km"

        # Horizontal mesh parameters
        NEX_XI = 80
        NEX_ETA = 80
        # Configure MPI decomposition
        if self.nproc == 32:
            NPROC_XI = 4
            NPROC_ETA = 8
        elif self.nproc == 16:
            NPROC_XI = 4
            NPROC_ETA = 4
        elif self.nproc == 8:
            NPROC_XI = 2
            NPROC_ETA = 4
        elif self.nproc == 4:
            NPROC_XI = 2
            NPROC_ETA = 2
        elif self.nproc == 1:  # For GPU mode
            NPROC_XI = 1
            NPROC_ETA = 1
        else:
            NPROC_XI = int(np.sqrt(self.nproc))
            NPROC_ETA = self.nproc // NPROC_XI
            assert NPROC_XI * NPROC_ETA == self.nproc, f"Unable to factorize {self.nproc} for NPROC_XI * NPROC_ETA"

        # Vertical mesh discretization: assign element counts per layer proportional to thickness
        total_vertical_elems = NEX_ZETA = 0
        vertical_elems = []
        for layer in layers:
            # Allocate elements in proportion to thickness (rounded to nearest integer)
            elems = int(round(layer["thickness"] / (domain_size[2]*1000) * NEX_XI))  # use NEX_XI as reference count
            vertical_elems.append(elems)
            NEX_ZETA += elems
        # Adjust if rounding caused mismatch in total:
        if NEX_ZETA != NEX_XI:
            vertical_elems[-1] += (NEX_XI - NEX_ZETA)
            NEX_ZETA = NEX_XI

        # Prepare material lines and region lines
        material_lines = []
        region_lines = []
        z_start = 1
        for i, layer in enumerate(layers, start=1):
            # Material line: material_id, rho, vp, vs, Qkappa, Qmu, anisotropy_flag, domain_id
            mat_id = i
            anisotropy_flag = 0  # 0 = no anisotropy
            domain_id = 2       # 2 = elastic solid
            m = layer
            material_lines.append(f"{mat_id} {m['rho']} {m['vp']} {m['vs']} {m['Qkappa']} {m['Qmu']} {anisotropy_flag} {domain_id}")
            # Region line: covers full X (1..NEX_XI) and Y (1..NEX_ETA), Z from z_start to z_end
            z_end = z_start + vertical_elems[i-1] - 1
            region_lines.append(f"1 {NEX_XI} 1 {NEX_ETA} {z_start} {z_end} {mat_id}")
            z_start = z_end + 1

        # Start writing Mesh_Par_file content
        mesh_par_content = []
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append("# Meshing input parameters")
        mesh_par_content.append("#-----------------------------------------------------------")
        # Domain coordinates (using them as Cartesian extents)
        mesh_par_content.append(f"LATITUDE_MIN = 0.0d0")
        mesh_par_content.append(f"LATITUDE_MAX = {domain_size[1]}d0")  # Y dimension in km
        mesh_par_content.append(f"LONGITUDE_MIN = 0.0d0")
        mesh_par_content.append(f"LONGITUDE_MAX = {domain_size[0]}d0")  # X dimension in km
        mesh_par_content.append(f"DEPTH_BLOCK_KM = {domain_size[2]}d0")  # Z dimension in km
        mesh_par_content.append(f"UTM_PROJECTION_ZONE = 13")  # just a valid UTM zone (not used when suppressed)
        mesh_par_content.append(f"SUPPRESS_UTM_PROJECTION = .true.")
        # Interface and cavity file names
        mesh_par_content.append(f"INTERFACES_FILE = interfaces.dat")
        mesh_par_content.append(f"CAVITY_FILE = no_cavity.dat")
        # Horizontal mesh and MPI
        mesh_par_content.append("# number of elements at the surface (must be multiple of NPROC)")
        mesh_par_content.append(f"NEX_XI = {NEX_XI}")
        mesh_par_content.append(f"NEX_ETA = {NEX_ETA}")
        mesh_par_content.append("# number of MPI processes along each horizontal direction")
        mesh_par_content.append(f"NPROC_XI = {NPROC_XI}")
        mesh_par_content.append(f"NPROC_ETA = {NPROC_ETA}")
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append("# Doubling layers (none for regular mesh)")
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append(f"USE_REGULAR_MESH = .true.")
        mesh_par_content.append(f"NDOUBLINGS = 0")
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append("# Visualization output")
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append(f"CREATE_ABAQUS_FILES = .false.")
        mesh_par_content.append(f"CREATE_DX_FILES = .false.")
        mesh_par_content.append(f"CREATE_VTK_FILES = .true.")
        mesh_par_content.append(f"LOCAL_PATH = ./DATABASES_MPI")
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append("# CPML absorbing boundaries")
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append(f"THICKNESS_OF_X_PML = {0.1 * domain_size[0]}d0")  # 10% of domain size
        mesh_par_content.append(f"THICKNESS_OF_Y_PML = {0.1 * domain_size[1]}d0")
        mesh_par_content.append(f"THICKNESS_OF_Z_PML = {0.1 * domain_size[2]}d0")
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append("# Domain materials")
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append(f"NMATERIALS = {len(material_lines)}")
        mesh_par_content.append("# material_id   rho   vp   vs   Q_kappa   Q_mu   anisotropy_flag   domain_id")
        mesh_par_content += material_lines
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append("# Domain regions")
        mesh_par_content.append("#-----------------------------------------------------------")
        mesh_par_content.append(f"NREGIONS = {len(region_lines)}")
        mesh_par_content.append("# XI_begin  XI_end   ETA_begin  ETA_end   Z_begin  Z_end   material_id")
        mesh_par_content += region_lines

        # Set output path if not provided
        if output_path is None:
            output_path = os.path.join(self.specfem_dir, "DATA", "meshfem3D_files", "Mesh_Par_file")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write file
        with open(output_path, "w") as f:
            f.write("\n".join(mesh_par_content) + "\n")
        
        logger.info(f"Generated Mesh_Par_file at {output_path}")
        logger.info(f"Regions: {region_lines}")
        
        # Generate interfaces.dat, interface files, and no_cavity.dat
        self._generate_interface_files(layers, domain_size, vertical_elems)
        
        return output_path
    
    def _generate_interface_files(self, layers, domain_size, vertical_elems):
        """Generate interface files needed by SPECFEM3D.
        
        Args:
            layers (list): Layer configuration
            domain_size (tuple): Domain size in km
            vertical_elems (list): Number of elements per layer
        """
        # Calculate interface depths (in meters)
        interface_depths = []
        current_depth = 0
        for i in range(len(layers) - 1):
            current_depth += layers[i]["thickness"]
            interface_depths.append(-current_depth)  # Negative for depth below surface
        
        # Create directory if it doesn't exist
        interface_dir = os.path.join(self.specfem_dir, "DATA", "meshfem3D_files")
        os.makedirs(interface_dir, exist_ok=True)
        
        # Generate interfaces.dat
        interfaces_lines = []
        interfaces_lines.append("# number of interfaces")
        interfaces_lines.append(f"{len(interface_depths)}")
        
        for i, depth in enumerate(interface_depths):
            interfaces_lines.append(f"# interface {i+1}")
            interfaces_lines.append(".true.")  # flat interface
            interfaces_lines.append(f"interface{i+1}.dat")
        
        interfaces_lines.append("# number of spectral elements per layer (from bottom)")
        for elem_count in vertical_elems:
            interfaces_lines.append(f"{elem_count}")
        
        with open(os.path.join(interface_dir, "interfaces.dat"), "w") as f:
            f.write("\n".join(interfaces_lines) + "\n")
        
        # Generate interface files (with constant depth)
        for i, depth in enumerate(interface_depths):
            with open(os.path.join(interface_dir, f"interface{i+1}.dat"), "w") as f:
                for _ in range(4):  # 4 corner points for a flat interface
                    f.write(f"{depth}\n")
        
        # Generate no_cavity.dat
        with open(os.path.join(interface_dir, "no_cavity.dat"), "w") as f:
            f.write("0\n")  # No cavity
        
        logger.info(f"Generated interfaces.dat with {len(interface_depths)} flat interfaces")
    
    def generate_par_file(self, simulation_time=2.0, dt=0.0005, source_position=(2000, 2000, 10), output_path=None):
        """Generate Par_file for SPECFEM3D.
        
        Args:
            simulation_time (float): Simulation time in seconds
            dt (float): Time step in seconds
            source_position (tuple): Source position (x, y, z) in meters
            output_path (str): Path to save the Par_file
            
        Returns:
            str: Path to the generated Par_file
        """
        # Calculate number of time steps
        nstep = int(simulation_time / dt)
        
        par_lines = []
        par_lines.append("#---------------------------- Simulation Parameters ----------------------------")
        par_lines.append(f"SIMULATION_TYPE = 1")
        par_lines.append(f"NOISE_TOMOGRAPHY = 0")
        par_lines.append(f"SAVE_FORWARD = .false.")
        par_lines.append(f"UTM_PROJECTION_ZONE = 13")
        par_lines.append(f"SUPPRESS_UTM_PROJECTION = .true.")
        par_lines.append(f"NPROC = {self.nproc}")
        par_lines.append(f"NSTEP = {nstep}")
        par_lines.append(f"DT = {dt}d0")
        par_lines.append(f"LTS_MODE = .false.")
        par_lines.append(f"PARTITIONING_TYPE = 1")
        par_lines.append(f"USE_LDDRK = .false.")
        par_lines.append(f"INCREASE_CFL_FOR_LDDRK = .false.")
        par_lines.append(f"RATIO_BY_WHICH_TO_INCREASE_IT = 1.0d0")
        par_lines.append(f"NGNOD = 8")
        par_lines.append(f"MODEL = default")
        par_lines.append(f"ATTENUATION = .false.")
        par_lines.append(f"ATTENUATION_f0_REFERENCE = 0.33333d0")
        par_lines.append(f"MIN_ATTENUATION_PERIOD = 999999998.d0")
        par_lines.append(f"MAX_ATTENUATION_PERIOD = 999999999.d0")
        par_lines.append(f"USE_OLSEN_ATTENUATION = .false.")
        par_lines.append(f"OLSEN_ATTENUATION_RATIO = 0.05")
        par_lines.append(f"STACEY_INSTEAD_OF_FREE_SURFACE = .false.")
        par_lines.append(f"PML_CONDITIONS = .true.")
        par_lines.append(f"PML_INSTEAD_OF_FREE_SURFACE = .false.")
        par_lines.append(f"ROTATE_PML_BOUNDARIES = .false.")
        par_lines.append(f"ADD_BASIN = .false.")
        par_lines.append(f"READ_EXTERNAL_SEM_MODEL = .false.")
        par_lines.append(f"INITIALIZE_GLMatNorms = .false.")
        par_lines.append("#---------------------------- Source Settings ----------------------------")
        par_lines.append(f"XSOURCE = {source_position[0]}")
        par_lines.append(f"YSOURCE = {source_position[1]}")
        par_lines.append(f"ZSOURCE = {source_position[2]}")
        par_lines.append(f"SOURCE_TYPE = 1")  # 1 for explosive, 2 for moment tensor
        par_lines.append(f"TIME_FUNCTION_TYPE = 1")  # 1 for Ricker wavelet
        par_lines.append(f"RECEIVERS_CAN_BE_BURIED = .false.")
        par_lines.append("#---------------------------- Output Settings ----------------------------")
        par_lines.append(f"USE_BINARY_FOR_SEISMOGRAMS = .false.")
        par_lines.append(f"SU_FORMAT = .false.")
        par_lines.append(f"NTSTEP_BETWEEN_OUTPUT_SEISMOS = 1")
        par_lines.append(f"NTSTEP_BETWEEN_OUTPUT_INFO = 1000")
        par_lines.append(f"OUTPUT_SEISMOS_ASCII_TEXT = .true.")
        par_lines.append(f"OUTPUT_SEISMOS_IN_SAC = .false.")
        par_lines.append(f"SAVE_SEISMOGRAMS_DISPLACEMENT = .true.")
        par_lines.append(f"SAVE_SEISMOGRAMS_VELOCITY = .false.")
        par_lines.append(f"SAVE_SEISMOGRAMS_ACCELERATION = .false.")
        par_lines.append(f"SAVE_SEISMOGRAMS_PRESSURE = .false.")
        par_lines.append(f"SAVE_SEISMOGRAMS_STRAIN = .false.")
        par_lines.append(f"WRITE_SEISMOGRAMS_BY_MPI = .false.")
        par_lines.append(f"GPU_MODE = {'.true.' if self.use_gpu else '.false.'}")
        par_lines.append(f"GPU_PLATFORM = 1")
        par_lines.append(f"GPU_DEVICE = 0")
        par_lines.append(f"PRINT_SOURCE_TIME_FUNCTION = .true.")
        par_lines.append(f"OUTPUT_SURFACE_MESH = .false.")
        par_lines.append(f"OUTPUT_VOLUME_MESH = .false.")
        par_lines.append(f"OUTPUT_VOLUME_HDF5 = .false.")
        par_lines.append(f"OUTPUT_SURFACE_HDF5 = .false.")

        # Set output path if not provided
        if output_path is None:
            output_path = os.path.join(self.specfem_dir, "DATA", "Par_file")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write file
        with open(output_path, "w") as f:
            f.write("\n".join(par_lines) + "\n")
        
        logger.info(f"Generated Par_file at {output_path} with NPROC={self.nproc}, NSTEP={nstep}, DT={dt}")
        
        return output_path
    
    def generate_station_file(self, stations=None, output_path=None):
        """Generate STATIONS file for SPECFEM3D.
        
        Args:
            stations (list): List of station configurations
                            Each station is a dict with keys: name, network, x, y, elevation, depth
            output_path (str): Path to save the STATIONS file
            
        Returns:
            str: Path to the generated STATIONS file
        """
        if not stations:
            # Default: create a grid of stations at the surface
            stations = []
            for i in range(5):
                for j in range(5):
                    x = 500 + i * 750  # From 500 to 3500 in 5 steps
                    y = 500 + j * 750  # From 500 to 3500 in 5 steps
                    stations.append({
                        "name": f"S{i+1}{j+1}",
                        "network": "XX",
                        "x": x,
                        "y": y,
                        "elevation": 0.0,
                        "depth": 0.0
                    })
        
        # Set output path if not provided
        if output_path is None:
            output_path = os.path.join(self.specfem_dir, "DATA", "STATIONS")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write station file
        station_lines = []
        for s in stations:
            station_lines.append(f"{s['name']}  {s['network']}  {s['x']}  {s['y']}  {s['elevation']}  {s['depth']}")
        
        with open(output_path, "w") as f:
            f.write("\n".join(station_lines) + "\n")
        
        logger.info(f"Generated STATIONS file at {output_path} with {len(stations)} stations")
        
        return output_path
    
    def generate_source_file(self, source_params=None, output_path=None):
        """Generate SOURCE file for SPECFEM3D.
        
        Args:
            source_params (dict): Source parameters
            output_path (str): Path to save the SOURCE file
            
        Returns:
            str: Path to the generated SOURCE file
        """
        # Default source parameters
        if source_params is None:
            source_params = {
                "xs": 2000.0,  # X position in m
                "ys": 2000.0,  # Y position in m
                "zs": 10.0,    # Z position in m (10 m below surface)
                "source_type": "explosive",  # 'explosive' or 'moment_tensor'
                "time_function_type": "ricker",  # 'ricker' or 'gaussian'
                "f0": 10.0,  # Dominant frequency in Hz
                "tshift": 0.0,  # Time shift in s
                "anglerota": 0.0,  # Rotation angle in degrees
                "anglerotb": 0.0,  # Rotation angle in degrees
            }
        
        # Set output path if not provided
        if output_path is None:
            output_path = os.path.join(self.specfem_dir, "DATA", "SOURCE")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Map source type to SPECFEM3D's numeric code
        source_type_map = {
            "explosive": 1,
            "moment_tensor": 2
        }
        source_type_code = source_type_map.get(source_params.get("source_type", "explosive"), 1)
        
        # Map time function type to SPECFEM3D's numeric code
        time_func_map = {
            "ricker": 1,
            "gaussian": 2
        }
        time_func_code = time_func_map.get(source_params.get("time_function_type", "ricker"), 1)
        
        # Write SOURCE file
        source_lines = []
        source_lines.append(f"source_surf                     = .false.")
        source_lines.append(f"xs                              = {source_params['xs']}")
        source_lines.append(f"ys                              = {source_params['ys']}")
        source_lines.append(f"zs                              = {source_params['zs']}")
        source_lines.append(f"source_type                     = {source_type_code}")
        source_lines.append(f"time_function_type              = {time_func_code}")
        source_lines.append(f"name_of_source_file             = CMTSOLUTION")
        source_lines.append(f"burst_band_width                = 0.0")
        source_lines.append(f"f0                              = {source_params['f0']}")
        source_lines.append(f"tshift                          = {source_params['tshift']}")
        source_lines.append(f"anglerota                       = {source_params['anglerota']}")
        source_lines.append(f"anglerotb                       = {source_params['anglerotb']}")
        
        # Add moment tensor components if needed
        if source_params.get("source_type", "explosive") == "moment_tensor":
            source_lines.append(f"Mxx                             = 1.0")
            source_lines.append(f"Myy                             = 1.0")
            source_lines.append(f"Mzz                             = 1.0")
            source_lines.append(f"Mxy                             = 0.0")
            source_lines.append(f"Mxz                             = 0.0")
            source_lines.append(f"Myz                             = 0.0")
        else:
            # For explosive source
            source_lines.append(f"Mxx                             = 1.0")
            source_lines.append(f"Myy                             = 1.0")
            source_lines.append(f"Mzz                             = 1.0")
            source_lines.append(f"Mxy                             = 0.0")
            source_lines.append(f"Mxz                             = 0.0")
            source_lines.append(f"Myz                             = 0.0")
        
        with open(output_path, "w") as f:
            f.write("\n".join(source_lines) + "\n")
        
        logger.info(f"Generated SOURCE file at {output_path} with f0={source_params['f0']} Hz")
        
        return output_path
    
    def setup_simulation(self, simulation_name, domain_size=(4.0, 4.0, 4.0), layers=None, 
                        simulation_time=2.0, dt=0.0005, source_position=(2000, 2000, 10),
                        source_freq=10.0, stations=None):
        """Set up a complete SPECFEM3D simulation.
        
        Args:
            simulation_name (str): Name of the simulation
            domain_size (tuple): Domain size in km (x, y, z)
            layers (list): Layer configurations
            simulation_time (float): Simulation duration in seconds
            dt (float): Time step in seconds
            source_position (tuple): Source position (x, y, z) in meters
            source_freq (float): Source dominant frequency in Hz
            stations (list): List of station configurations
            
        Returns:
            dict: Paths to generated files
        """
        # Create simulation directory
        sim_dir = os.path.join(self.output_dir, "raw", simulation_name)
        os.makedirs(sim_dir, exist_ok=True)
        
        # Generate Mesh_Par_file
        mesh_par_path = os.path.join(sim_dir, "Mesh_Par_file")
        self.generate_mesh_par_file(domain_size=domain_size, layers=layers, output_path=mesh_par_path)
        
        # Generate Par_file
        par_path = os.path.join(sim_dir, "Par_file")
        self.generate_par_file(simulation_time=simulation_time, dt=dt, 
                              source_position=source_position, output_path=par_path)
        
        # Generate STATIONS file
        stations_path = os.path.join(sim_dir, "STATIONS")
        self.generate_station_file(stations=stations, output_path=stations_path)
        
        # Generate SOURCE file
        source_path = os.path.join(sim_dir, "SOURCE")
        source_params = {
            "xs": source_position[0],
            "ys": source_position[1],
            "zs": source_position[2],
            "source_type": "explosive",
            "time_function_type": "ricker",
            "f0": source_freq,
            "tshift": 0.0,
            "anglerota": 0.0,
            "anglerotb": 0.0
        }
        self.generate_source_file(source_params=source_params, output_path=source_path)
        
        # Copy interface files
        interface_dir = os.path.join(self.specfem_dir, "DATA", "meshfem3D_files")
        for file in ["interfaces.dat", "no_cavity.dat"] + [f"interface{i+1}.dat" for i in range(len(layers)-1)]:
            src = os.path.join(interface_dir, file)
            dst = os.path.join(sim_dir, file)
            if os.path.exists(src):
                shutil.copy(src, dst)
        
        logger.info(f"Set up simulation {simulation_name} in {sim_dir}")
        
        return {
            "mesh_par_file": mesh_par_path,
            "par_file": par_path,
            "stations_file": stations_path,
            "source_file": source_path,
            "simulation_dir": sim_dir
        }
    
    def run_simulation(self, simulation_dir, clean=True):
        """Run a SPECFEM3D simulation.
        
        Args:
            simulation_dir (str): Path to the simulation directory
            clean (bool): Whether to clean up after running
            
        Returns:
            bool: Whether the simulation completed successfully
        """
        # Copy simulation files to SPECFEM3D DATA directory
        for file in ["Par_file", "SOURCE", "STATIONS"]:
            src = os.path.join(simulation_dir, file)
            dst = os.path.join(self.specfem_dir, "DATA", file)
            if os.path.exists(src):
                shutil.copy(src, dst)
        
        # Copy mesh files
        mesh_dir = os.path.join(self.specfem_dir, "DATA", "meshfem3D_files")
        os.makedirs(mesh_dir, exist_ok=True)
        for file in ["Mesh_Par_file", "interfaces.dat", "no_cavity.dat"]:
            src = os.path.join(simulation_dir, file)
            dst = os.path.join(mesh_dir, file)
            if os.path.exists(src):
                shutil.copy(src, dst)
        
        # Copy interface files
        for file in os.listdir(simulation_dir):
            if file.startswith("interface") and file.endswith(".dat"):
                src = os.path.join(simulation_dir, file)
                dst = os.path.join(mesh_dir, file)
                shutil.copy(src, dst)
        
        # Change to SPECFEM3D directory
        cwd = os.getcwd()
        os.chdir(self.specfem_dir)
        
        try:
            # Run mesher
            logger.info("Running xmeshfem3D...")
            result = subprocess.run(["mpirun", "-np", str(self.nproc), "./bin/xmeshfem3D"], 
                                   check=True, capture_output=True, text=True)
            logger.info("xmeshfem3D completed successfully")
            
            # Run database generator
            logger.info("Running xgenerate_databases...")
            result = subprocess.run(["mpirun", "-np", str(self.nproc), "./bin/xgenerate_databases"],
                                  check=True, capture_output=True, text=True)
            logger.info("xgenerate_databases completed successfully")
            
            # Run solver
            logger.info("Running xspecfem3D...")
            result = subprocess.run(["mpirun", "-np", str(self.nproc), "./bin/xspecfem3D"],
                                  check=True, capture_output=True, text=True)
            logger.info("xspecfem3D completed successfully")
            
            # Copy output files back to simulation directory
            output_dir = os.path.join(simulation_dir, "OUTPUT_FILES")
            os.makedirs(output_dir, exist_ok=True)
            for file in os.listdir("OUTPUT_FILES"):
                src = os.path.join("OUTPUT_FILES", file)
                dst = os.path.join(output_dir, file)
                if os.path.isfile(src):
                    shutil.copy(src, dst)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Simulation failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
        
        finally:
            # Change back to original directory
            os.chdir(cwd)
            
            # Clean up if requested
            if clean:
                logger.info("Cleaning up...")
                for dir_to_clean in ["DATABASES_MPI", "LOCAL_PATH"]:
                    dir_path = os.path.join(self.specfem_dir, dir_to_clean)
                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)
    
    def prepare_parfile(self, template_path, params):
        """
        Prepare Par_file from template and simulation parameters.
        
        Args:
            template_path (str): Path to Par_file template
            params (dict): Simulation parameters
        
        Returns:
            str: Path to the prepared Par_file
        """
        # Set NPROC = number of processors
        nproc = params.get('NPROC', 4)
        self.nproc = nproc
        print(f"Set NPROC to {nproc}")
        
        # Ensure NPROC_XI and NPROC_ETA are set correctly
        if 'NPROC_XI' not in params or 'NPROC_ETA' not in params:
            # Default to 2x2 configuration
            params['NPROC_XI'] = 2
            params['NPROC_ETA'] = 2
        
        # Check that NPROC = NPROC_XI * NPROC_ETA
        if params['NPROC_XI'] * params['NPROC_ETA'] != nproc:
            logger.warning(f"NPROC ({nproc}) doesn't match NPROC_XI*NPROC_ETA ({params['NPROC_XI']*params['NPROC_ETA']})")
            # Fix NPROC_XI and NPROC_ETA
            if nproc == 1:
                params['NPROC_XI'] = 1
                params['NPROC_ETA'] = 1
            elif nproc == 4:
                params['NPROC_XI'] = 2
                params['NPROC_ETA'] = 2
            else:
                # Try to find a reasonable factorization
                for xi in range(1, nproc+1):
                    if nproc % xi == 0:
                        eta = nproc // xi
                        if abs(xi - eta) < abs(params['NPROC_XI'] - params['NPROC_ETA']):
                            params['NPROC_XI'] = xi
                            params['NPROC_ETA'] = eta
        
        # Read the template
        try:
            with open(template_path, 'r') as f:
                template_content = f.read()
        except FileNotFoundError:
            logger.error(f"Par_file template not found at {template_path}")
            # Create a minimal template if not found
            template_content = "# Minimal SPECFEM3D Par_file\n"
            logger.warning("Created minimal Par_file template")
        
        # Update parameters in template
        for param, value in params.items():
            # Skip some parameters that users may add that don't belong in Par_file
            if param in ['mpirun', 'np', 'cmd']:
                continue
            
            # Check if parameter exists in template
            param_pattern = re.compile(rf"^{param}\s*=.*$", re.MULTILINE)
            if param_pattern.search(template_content):
                # Update existing parameter
                template_content = param_pattern.sub(f"{param:30s} = {value}", template_content)
            else:
                # Parameter doesn't exist, log a warning
                logger.warning(f"Parameter '{param}' not found in Par_file template. Skipping.")
                print(f"Warning: Parameter '{param}' not found in Par_file template. Skipping.")
        
        # Check for missing required parameters
        required_params = [
            'NPROC', 'MODEL', 'NGNOD', 'LATITUDE_MIN', 'LATITUDE_MAX', 
            'LONGITUDE_MIN', 'LONGITUDE_MAX', 'DEPTH_MIN', 'DEPTH_MAX', 
            'NEX_XI', 'NEX_ETA', 'NPROC_XI', 'NPROC_ETA', 'NUMBER_OF_SIMULTANEOUS_RUNS',
            'BROADCAST_SAME_MESH_AND_MODEL'
        ]
        
        missing_params = []
        for param in required_params:
            param_pattern = re.compile(rf"^{param}\s*=.*$", re.MULTILINE)
            if not param_pattern.search(template_content):
                missing_params.append(param)
        
        if missing_params:
            logger.warning(f"Par_file is missing required parameters: {', '.join(missing_params)}")
            print(f"Warning: Par_file is missing required parameters: {', '.join(missing_params)}")
            
            # Add the missing parameters using values from params dict if available
            for param in missing_params:
                if param in params:
                    template_content += f"\n{param:30s} = {params[param]}"
                else:
                    # Use defaults for required parameters not provided
                    defaults = {
                        'NPROC': 4,
                        'MODEL': 'default',
                        'NGNOD': 8,
                        'LATITUDE_MIN': 0.0,
                        'LATITUDE_MAX': 2000.0,
                        'LONGITUDE_MIN': 0.0,
                        'LONGITUDE_MAX': 2000.0,
                        'DEPTH_MIN': 0.0,
                        'DEPTH_MAX': 2000.0,
                        'NEX_XI': 40,
                        'NEX_ETA': 40,
                        'NPROC_XI': 2,
                        'NPROC_ETA': 2,
                        'NUMBER_OF_SIMULTANEOUS_RUNS': 1,
                        'BROADCAST_SAME_MESH_AND_MODEL': '.true.'
                    }
                    template_content += f"\n{param:30s} = {defaults.get(param, '')}"
        
        # Write the prepared Par_file
        par_file_path = os.path.join(self.output_dir, "Par_file")
        with open(par_file_path, 'w') as f:
            f.write(template_content)
        
        logger.info(f"Prepared Par_file at {par_file_path}")
        return par_file_path
    
    def prepare_source(self, source_params):
        """
        Prepare a SOURCE file for SPECFEM3D simulation.
        
        Args:
            source_params (dict): Source parameters
            
        Returns:
            str: Path to the prepared SOURCE file
        """
        # Template for SOURCE file content
        source_template = """
        source_surf = {source_surf}
        xs = {xs}
        ys = {ys}
        zs = {zs}
        source_type = {source_type}
        time_function_type = {time_function_type}
        name_of_source_file = {name_of_source_file}
        burst_band_width = {burst_band_width}
        f0 = {f0}
        tshift = {tshift}
        anglesource = {anglesource}
        Mxx = {Mxx}
        Mxy = {Mxy}
        Mxz = {Mxz}
        Myy = {Myy}
        Myz = {Myz}
        Mzz = {Mzz}
        """
        
        # Fill in the template with source parameters
        source_content = source_template.format(**source_params)
        
        # Write the SOURCE file
        output_source_file = self.output_dir / "SOURCE"
        with open(output_source_file, 'w') as f:
            f.write(source_content)
            
        logger.info(f"Prepared SOURCE file at {output_source_file}")
        return str(output_source_file)
    
    def prepare_stations(self, stations_list):
        """
        Prepare a STATIONS file for receivers.
        
        Args:
            stations_list (list): List of station dictionaries with name, network, lat, lon, elevation, burial
            
        Returns:
            str: Path to the prepared STATIONS file
        """
        stations_content = ""
        for station in stations_list:
            # Format: STA NET LAT LON ELEVATION BURIAL
            line = f"{station['name']} {station['network']} {station['lat']} {station['lon']} {station['elevation']} {station['burial']}\n"
            stations_content += line
            
        # Write the STATIONS file
        output_stations_file = self.output_dir / "STATIONS"
        with open(output_stations_file, 'w') as f:
            f.write(stations_content)
            
        logger.info(f"Prepared STATIONS file with {len(stations_list)} stations")
        return str(output_stations_file)
    
    def run_mesher(self):
        """
        Run the SPECFEM3D mesher (xmeshfem3D).
        
        Returns:
            bool: True if meshing completes successfully
        """
        # Create necessary directories
        data_dir = self.specfem_dir / "DATA"
        mesh_dir = data_dir / "meshfem3D_files"
        os.makedirs(mesh_dir, exist_ok=True)
        
        # Copy Par_file to DATA directory
        shutil.copy(self.output_dir / "Par_file", data_dir / "Par_file")
        
        # Copy SOURCE and STATIONS if they exist
        if (self.output_dir / "SOURCE").exists():
            shutil.copy(self.output_dir / "SOURCE", data_dir / "SOURCE")
        
        if (self.output_dir / "STATIONS").exists():
            shutil.copy(self.output_dir / "STATIONS", data_dir / "STATIONS")
        
        # Change to SPECFEM directory and run mesher
        cwd = os.getcwd()
        os.chdir(self.specfem_dir)
        
        try:
            print("Running xmeshfem3D...")
            # Use the same number of MPI processes as specified in Par_file
            cmd = f"mpirun -np {self.nproc} ./bin/xmeshfem3D"
                
            print(f"Executing: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Meshing completed successfully")
            print("Meshing completed successfully")
            os.chdir(cwd)
            return True
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Meshing failed: {error_message}")
            print(f"Meshing failed: {error_message}")
            os.chdir(cwd)
            return False
    
    def run_database_generator(self):
        """
        Run the SPECFEM3D database generator (xgenerate_databases).
        
        Returns:
            bool: True if database generation completes successfully
        """
        # Change to SPECFEM directory
        cwd = os.getcwd()
        os.chdir(self.specfem_dir)
        
        try:
            logger.info("Running xgenerate_databases...")
            print("Running xgenerate_databases...")
            cmd = f"mpirun -np {self.nproc} ./bin/xgenerate_databases"
                
            print(f"Executing: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Database generation completed successfully")
            print("Database generation completed successfully")
            os.chdir(cwd)
            return True
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Database generation failed: {error_message}")
            print(f"Database generation failed: {error_message}")
            os.chdir(cwd)
            return False
    
    def run_solver(self):
        """
        Run the SPECFEM3D solver (xspecfem3D).
        
        Returns:
            bool: True if simulation completes successfully
        """
        # Change to SPECFEM directory and run solver
        cwd = os.getcwd()
        os.chdir(self.specfem_dir)
        
        try:
            logger.info("Running xspecfem3D...")
            print("Running xspecfem3D...")
            # Use the same number of MPI processes as specified in Par_file
            cmd = f"mpirun -np {self.nproc} ./bin/xspecfem3D"
                
            print(f"Executing: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Simulation completed successfully")
            print("Simulation completed successfully")
            
            # Copy output files to output_dir
            output_path = self.specfem_dir / "OUTPUT_FILES"
            if output_path.exists():
                print(f"Copying output files from {output_path} to {self.output_dir}...")
                file_count = 0
                for file in output_path.glob("*"):
                    shutil.copy(file, self.output_dir)
                    file_count += 1
                logger.info(f"Copied {file_count} output files to {self.output_dir}")
                print(f"Copied {file_count} output files to {self.output_dir}")
            else:
                print(f"Warning: OUTPUT_FILES directory not found at {output_path}")
            
            os.chdir(cwd)
            return True
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Simulation failed: {error_message}")
            print(f"Simulation failed: {error_message}")
            os.chdir(cwd)
            return False
    
    def clean_previous_outputs(self):
        """
        Clean previous simulation outputs to avoid conflicts, especially
        with different NPROC settings from previous runs.
        """
        print("Cleaning previous simulation outputs...")
        
        # Change to SPECFEM directory
        cwd = os.getcwd()
        os.chdir(self.specfem_dir)
        
        # Clean OUTPUT_FILES directory
        output_path = self.specfem_dir / "OUTPUT_FILES"
        if output_path.exists():
            try:
                # Define patterns of files to remove
                patterns_to_remove = [
                    "*.bin",        # Binary files
                    "*.sem*",       # Seismogram files
                    "error_*",      # Error files
                    "timestamp*",   # Timestamp files
                    "output_*",     # Output log files
                    "DB.*",         # Database files
                    "proc*",        # Processor-specific files
                    "surface_from_mesher.h", # Mesh descriptor file
                    "values_from_mesher.h"   # Mesh values file
                ]
                
                # Remove files matching patterns
                for pattern in patterns_to_remove:
                    for item in output_path.glob(pattern):
                        try:
                            if item.is_file():
                                item.unlink()
                                print(f"Removed file: {item.name}")
                            elif item.is_dir():
                                shutil.rmtree(item)
                                print(f"Removed directory: {item.name}")
                        except Exception as e:
                            print(f"Warning: Could not remove {item}: {e}")
                
                # Handle DATABASES_MPI directory specially
                db_mpi_dir = output_path / "DATABASES_MPI"
                if db_mpi_dir.exists():
                    try:
                        # Remove all content but keep the directory
                        for item in db_mpi_dir.glob("*"):
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                        print(f"Cleaned DATABASES_MPI directory")
                    except Exception as e:
                        print(f"Warning: Error cleaning DATABASES_MPI directory: {e}")
                
                print(f"Cleaned OUTPUT_FILES directory")
            except Exception as e:
                print(f"Warning: Error cleaning OUTPUT_FILES: {e}")
        else:
            # Create OUTPUT_FILES directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Created OUTPUT_FILES directory")
            
            # Create DATABASES_MPI directory if it doesn't exist
            db_mpi_dir = output_path / "DATABASES_MPI"
            db_mpi_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created DATABASES_MPI directory")
        
        # Return to original directory
        os.chdir(cwd)
    
    def run_full_simulation(self, par_file_template=None, simulation_params=None, source_params=None, stations_list=None, clean_outputs=True):
        """
        Run a complete SPECFEM3D simulation (mesh, partition, solve).
        
        Args:
            par_file_template (str): Path to Par_file template
            simulation_params (dict): Parameters to modify in the Par_file
            source_params (dict, optional): Source parameters
            stations_list (list, optional): List of station dictionaries
            clean_outputs (bool, optional): Whether to clean previous outputs before running
            
        Returns:
            bool: True if simulation completes successfully
        """
        # Clean previous outputs if requested
        if clean_outputs:
            self.clean_previous_outputs()
            
        # Generate parameters if template and params are provided
        if par_file_template and simulation_params:
            self.prepare_parfile(par_file_template, simulation_params)
        
        # Generate mesh parameters if not done through template
        if not os.path.exists(os.path.join(self.specfem_dir, "DATA", "meshfem3D_files", "Mesh_Par_file")):
            self.generate_mesh_par_file()
        
        # Generate Par_file if not done through template
        if not os.path.exists(os.path.join(self.specfem_dir, "DATA", "Par_file")):
            self.generate_par_file()
        
        # Generate source file
        if source_params:
            self.prepare_source(source_params)
        elif not os.path.exists(os.path.join(self.specfem_dir, "DATA", "SOURCE")):
            self.generate_source_file()
            
        # Generate stations file
        if stations_list:
            self.prepare_stations(stations_list)
        elif not os.path.exists(os.path.join(self.specfem_dir, "DATA", "STATIONS")):
            self.generate_station_file()
        
        # Run the mesher
        if not self.run_mesher():
            return False
        
        # Run the database generator
        if not self.run_database_generator():
            return False
        
        # Run the solver
        return self.run_solver()