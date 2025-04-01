#!/usr/bin/env python3
"""
Script to run SPECFEM3D simulations with proper parameter file generation.

This script demonstrates how to use the SpecfemSimulation class to generate
SPECFEM3D parameter files and run simulations with layered velocity models.
"""
import os
import argparse
import logging
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
    args = parser.parse_args()

    # Create simulation handler
    sim = SpecfemSimulation(
        specfem_dir=args.specfem_dir,
        output_dir=args.output_dir,
        nproc=args.nproc,
        use_gpu=args.use_gpu
    )

    # Define 3-layer velocity model (from bottom to top)
    layers = [
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

    # Set up the simulation
    source_position = (args.source_x, args.source_y, args.source_z)
    simulation_files = sim.setup_simulation(
        simulation_name=args.simulation_name,
        domain_size=domain_size,
        layers=layers,
        simulation_time=args.simulation_time,
        dt=args.dt,
        source_position=source_position,
        source_freq=args.source_freq,
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