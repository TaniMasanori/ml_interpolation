#!/usr/bin/env python3
"""
Script to run SPECFEM3D simulations with proper parameter file generation.

This script demonstrates how to use the SpecfemSimulation class to generate
SPECFEM3D parameter files and run simulations with layered velocity models.
"""
import os
import argparse
import logging
import json
from src.simulation.specfem_runner import SpecfemSimulation
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('specfem_simulation.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run SPECFEM3D simulation with proper parameter files."""
    parser = argparse.ArgumentParser(description='Run SPECFEM3D simulation with proper parameter files.')
    parser.add_argument('--specfem_dir', type=str, default=os.path.expanduser("~/specfem3d"), 
                        help='Path to SPECFEM3D installation')
    parser.add_argument('--output_dir', type=str, default='data/synthetic', 
                        help='Path to store simulation outputs')
    parser.add_argument('--simulation_name', type=str, default='simulation1', help='Name of the simulation')
    parser.add_argument('--nproc', type=int, default=4, help='Number of MPI processes')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--domain_size_x', type=float, default=4.0, help='Domain size in X direction (km)')
    parser.add_argument('--domain_size_y', type=float, default=4.0, help='Domain size in Y direction (km)')
    parser.add_argument('--domain_size_z', type=float, default=4.0, help='Domain size in Z direction (km)')
    parser.add_argument('--simulation_time', type=float, default=2.0, help='Simulation time (s)')
    parser.add_argument('--dt', type=float, default=0.0005, help='Time step (s)')
    parser.add_argument('--source_x', type=float, default=2000.0, help='Source X position (m)')
    parser.add_argument('--source_y', type=float, default=2000.0, help='Source Y position (m)')
    parser.add_argument('--source_z', type=float, default=10.0, help='Source Z position (m)')
    parser.add_argument('--source_freq', type=float, default=10.0, help='Source dominant frequency (Hz)')
    parser.add_argument('--setup_only', action='store_true', help='Only set up the simulation, do not run it')
    parser.add_argument('--parameter_file', type=str, help='Path to JSON parameter file')
    args = parser.parse_args()

    # Create simulation handler
    sim = SpecfemSimulation(
        specfem_dir=args.specfem_dir,
        output_dir=args.output_dir,
        nproc=args.nproc,
        use_gpu=args.use_gpu
    )

    # If a parameter file is provided, use it to configure the simulation
    if args.parameter_file:
        logger.info(f"Loading parameters from: {args.parameter_file}")
        try:
            with open(args.parameter_file, 'r') as f:
                params = json.load(f)
            
            # Extract model layers if available
            if 'model' in params and 'layers' in params['model']:
                layers = params['model']['layers']
                logger.info(f"Using velocity model with {len(layers)} layers")
                
                # Convert layers to the format expected by SpecfemSimulation
                sim_layers = []
                for layer in layers:
                    sim_layer = {
                        "thickness": layer['thickness'] * 1000.0,  # convert to meters
                        "rho": layer['rho'],
                        "vp": layer['vp'] * 1000.0,  # convert to m/s
                        "vs": layer['vs'] * 1000.0,  # convert to m/s
                        "Qkappa": layer.get('q', 9999.0),
                        "Qmu": layer.get('q', 9999.0)
                    }
                    sim_layers.append(sim_layer)
            else:
                # Define default 3-layer velocity model if not in parameter file
                sim_layers = [
                    {
                        "thickness": 1000.0,  # 1 km in meters
                        "rho": 2500.0,        # density in kg/m^3
                        "vp": 3500.0,         # P-wave velocity in m/s
                        "vs": 2000.0,         # S-wave velocity in m/s
                        "Qkappa": 9999.0,     # Quality factor for P-waves (high = no attenuation)
                        "Qmu": 9999.0         # Quality factor for S-waves
                    },
                    {
                        "thickness": 2000.0,  # 2 km in meters
                        "rho": 2200.0,
                        "vp": 2500.0,
                        "vs": 1200.0,
                        "Qkappa": 9999.0,
                        "Qmu": 9999.0
                    },
                    {
                        "thickness": 1000.0,  # 1 km in meters
                        "rho": 2000.0,
                        "vp": 1500.0,
                        "vs": 500.0,
                        "Qkappa": 9999.0,
                        "Qmu": 9999.0
                    }
                ]
                logger.info("No velocity model found in parameter file, using default layers")
            
            # Extract stations if available
            if 'stations' in params:
                stations = params['stations']
                logger.info(f"Using {len(stations)} stations from parameter file")
                
                # Convert station coordinates from lat/lon to x/y if needed
                converted_stations = []
                for station in stations:
                    # Check if x/y coordinates are already present
                    if 'x' in station and 'y' in station:
                        converted_stations.append(station)
                    # Otherwise, convert lat/lon to x/y
                    elif 'lat' in station and 'lon' in station:
                        converted_station = {
                            "name": station['name'],
                            "network": station['network'],
                            "x": station['lat'],
                            "y": station['lon'],
                            "elevation": station.get('elevation', 0.0),
                            "depth": station.get('burial', 0.0)
                        }
                        converted_stations.append(converted_station)
                
                stations = converted_stations
            else:
                # Set up a regular grid of 25 stations at the surface (5x5 grid)
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
                logger.info("No stations found in parameter file, using default 5x5 grid")
            
            # Extract source parameters if available
            if 'source' in params:
                source = params['source']
                source_position = (
                    source.get('latorUTM', args.source_x),
                    source.get('longorUTM', args.source_y),
                    source.get('depth', args.source_z)
                )
                source_freq = 10.0  # Default frequency
                logger.info(f"Using source from parameter file at: {source_position}")
            else:
                source_position = (args.source_x, args.source_y, args.source_z)
                source_freq = args.source_freq
                logger.info(f"No source found in parameter file, using default")
            
            # Extract simulation parameters
            simulation_time = args.simulation_time
            dt = args.dt
            if 'simulation' in params:
                sim_params = params['simulation']
                if 'dt' in sim_params:
                    dt = sim_params['dt']
                if 'nt' in sim_params:
                    simulation_time = sim_params['nt'] * dt
                if 'source' in sim_params and 'frequency' in sim_params['source']:
                    source_freq = sim_params['source']['frequency']
            
            domain_size = (args.domain_size_x, args.domain_size_y, args.domain_size_z)
            
        except Exception as e:
            logger.error(f"Error loading parameter file: {str(e)}")
            sys.exit(1)
    else:
        # Define 3-layer velocity model (from bottom to top)
        sim_layers = [
            # Bottom layer: 1 km thick, high velocity
            {
                "thickness": 1000.0,  # 1 km in meters
                "rho": 2500.0,        # density in kg/m^3
                "vp": 3500.0,         # P-wave velocity in m/s
                "vs": 2000.0,         # S-wave velocity in m/s
                "Qkappa": 9999.0,     # Quality factor for P-waves (high = no attenuation)
                "Qmu": 9999.0         # Quality factor for S-waves
            },
            # Middle layer: 2 km thick, intermediate velocity
            {
                "thickness": 2000.0,  # 2 km in meters
                "rho": 2200.0,
                "vp": 2500.0,
                "vs": 1200.0,
                "Qkappa": 9999.0,
                "Qmu": 9999.0
            },
            # Top layer: 1 km thick, low velocity
            {
                "thickness": 1000.0,  # 1 km in meters
                "rho": 2000.0,
                "vp": 1500.0,
                "vs": 500.0,
                "Qkappa": 9999.0,
                "Qmu": 9999.0
            }
        ]

        # Define domain size in km
        domain_size = (args.domain_size_x, args.domain_size_y, args.domain_size_z)

        # Set up a regular grid of 25 stations at the surface (5x5 grid)
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

        source_position = (args.source_x, args.source_y, args.source_z)
        source_freq = args.source_freq
        simulation_time = args.simulation_time
        dt = args.dt

    # Set up the simulation
    simulation_files = sim.setup_simulation(
        simulation_name=args.simulation_name,
        domain_size=domain_size,
        layers=sim_layers,
        simulation_time=simulation_time,
        dt=dt,
        source_position=source_position,
        source_freq=source_freq,
        stations=stations
    )

    logger.info(f"Simulation files prepared in {simulation_files['simulation_dir']}")
    print(f"Simulation files prepared in {simulation_files['simulation_dir']}")

    # Run the simulation if not setup only
    if not args.setup_only:
        logger.info("Running the SPECFEM3D simulation...")
        print("Running the SPECFEM3D simulation...")
        success = sim.run_simulation(simulation_files['simulation_dir'])
        if success:
            logger.info("Simulation completed successfully.")
            print("\n=== Simulation Completed Successfully ===")
            print(f"Output files are available in:")
            print(f"- Simulation directory: {simulation_files['simulation_dir']}")
            print(f"- Output directory: {simulation_files['simulation_dir']}/OUTPUT_FILES")
            print("\nNext steps:")
            print("1. Run DAS conversion: python convert_seismo_to_das.py")
            print("2. Process data: python notebooks/03_data_preprocessing.ipynb")
            print("3. Train model: python notebooks/04_model_training.ipynb")
        else:
            logger.error("Simulation failed.")
            print("\n=== Simulation Failed ===")
            print("Please check the specfem_simulation.log file for details.")
    else:
        logger.info("Setup complete. Use the following command to run the simulation:")
        cmd = f"python run_specfem.py --specfem_dir {args.specfem_dir} --output_dir {args.output_dir} --simulation_name {args.simulation_name} --nproc {args.nproc}{' --use_gpu' if args.use_gpu else ''}"
        logger.info(cmd)
        print("\n=== Simulation Setup Complete ===")
        print("Use the following command to run the simulation:")
        print(cmd)

if __name__ == "__main__":
    main()