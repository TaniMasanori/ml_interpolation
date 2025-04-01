#!/usr/bin/env python3
"""
SPECFEM3D Parameter Manager

This script provides utilities to manage parameter files for SPECFEM3D simulations:
- Par_file: Main simulation parameters
- Mesh_Par_file: Mesh generation parameters
- CMTSOLUTION: Source parameters
- STATIONS: Receiver locations

Features:
- Load and parse parameter files
- Modify parameters programmatically
- Save updated parameter files
- Create parameter templates for different simulation scenarios
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

class ParamManager:
    """Manage SPECFEM3D parameter files."""
    
    def __init__(self, specfem_dir=None, data_dir=None):
        """
        Initialize the parameter manager.
        
        Args:
            specfem_dir (str): Path to SPECFEM3D installation directory
            data_dir (str): Path to data directory containing parameter files
        """
        self.specfem_dir = os.path.expanduser(specfem_dir or "~/specfem3d")
        self.data_dir = data_dir or os.path.join(self.specfem_dir, "DATA")
        
        # Parameter file paths
        self.par_file_path = os.path.join(self.data_dir, "Par_file")
        self.mesh_par_file_path = os.path.join(self.data_dir, "meshfem3D_files", "Mesh_Par_file")
        self.source_file_path = os.path.join(self.data_dir, "CMTSOLUTION")
        self.stations_file_path = os.path.join(self.data_dir, "STATIONS")
        
        # Parameter storage
        self.par_file_params = {}
        self.mesh_par_file_params = {}
        self.source_params = {}
        self.stations = []
        
        # Load parameters if files exist
        if os.path.exists(self.par_file_path):
            self.load_par_file()
        
        if os.path.exists(self.mesh_par_file_path):
            self.load_mesh_par_file()
        
        if os.path.exists(self.source_file_path):
            self.load_source_file()
        
        if os.path.exists(self.stations_file_path):
            self.load_stations_file()
    
    def load_par_file(self, file_path=None):
        """
        Load parameters from Par_file.
        
        Args:
            file_path (str, optional): Override default Par_file path
            
        Returns:
            dict: Loaded parameters
        """
        file_path = file_path or self.par_file_path
        
        if not os.path.exists(file_path):
            print(f"Warning: Par_file not found at {file_path}")
            return {}
        
        params = {}
        with open(file_path, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.strip().startswith('#') or not line.strip():
                    continue
                
                # Extract parameter name and value
                if '=' in line:
                    parts = line.split('=', 1)
                    param_name = parts[0].strip()
                    param_value = parts[1].strip()
                    
                    # Remove any trailing comments
                    if '#' in param_value:
                        param_value = param_value.split('#', 1)[0].strip()
                    
                    # Convert to appropriate type if possible
                    if param_value.isdigit():
                        param_value = int(param_value)
                    elif self._is_float(param_value):
                        param_value = float(param_value)
                    elif param_value.lower() in ['.true.', '.false.']:
                        param_value = param_value.lower() == '.true.'
                    
                    params[param_name] = param_value
        
        self.par_file_params = params
        return params
    
    def load_mesh_par_file(self, file_path=None):
        """
        Load parameters from Mesh_Par_file.
        
        Args:
            file_path (str, optional): Override default Mesh_Par_file path
            
        Returns:
            dict: Loaded parameters
        """
        file_path = file_path or self.mesh_par_file_path
        
        if not os.path.exists(file_path):
            print(f"Warning: Mesh_Par_file not found at {file_path}")
            return {}
        
        # Mesh_Par_file has same format as Par_file
        params = {}
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip().startswith('#') or not line.strip():
                    continue
                
                if '=' in line:
                    parts = line.split('=', 1)
                    param_name = parts[0].strip()
                    param_value = parts[1].strip()
                    
                    if '#' in param_value:
                        param_value = param_value.split('#', 1)[0].strip()
                    
                    if param_value.isdigit():
                        param_value = int(param_value)
                    elif self._is_float(param_value):
                        param_value = float(param_value)
                    elif param_value.lower() in ['.true.', '.false.']:
                        param_value = param_value.lower() == '.true.'
                    
                    params[param_name] = param_value
        
        self.mesh_par_file_params = params
        return params
    
    def load_source_file(self, file_path=None):
        """
        Load parameters from CMTSOLUTION source file.
        
        Args:
            file_path (str, optional): Override default source file path
            
        Returns:
            dict: Loaded source parameters
        """
        file_path = file_path or self.source_file_path
        
        if not os.path.exists(file_path):
            print(f"Warning: Source file not found at {file_path}")
            return {}
        
        params = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Parse header line
            header = lines[0].strip()
            params['event_name'] = header
            
            # Parse other parameters
            for line in lines[1:]:
                if ':' in line:
                    parts = line.split(':', 1)
                    param_name = parts[0].strip()
                    param_value = parts[1].strip()
                    
                    # Convert to appropriate type if possible
                    if self._is_float(param_value):
                        param_value = float(param_value)
                    
                    params[param_name] = param_value
        
        self.source_params = params
        return params
    
    def load_stations_file(self, file_path=None):
        """
        Load station information from STATIONS file.
        
        Args:
            file_path (str, optional): Override default stations file path
            
        Returns:
            list: List of station dictionaries
        """
        file_path = file_path or self.stations_file_path
        
        if not os.path.exists(file_path):
            print(f"Warning: Stations file not found at {file_path}")
            return []
        
        stations = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        station = {
                            'name': parts[0],
                            'network': parts[1],
                            'lat': float(parts[2]),
                            'lon': float(parts[3]),
                            'elevation': float(parts[4]),
                            'burial': float(parts[5])
                        }
                        stations.append(station)
        
        self.stations = stations
        return stations
    
    def save_par_file(self, output_path=None, params=None):
        """
        Save parameters to Par_file.
        
        Args:
            output_path (str, optional): Override default Par_file path
            params (dict, optional): Parameters to save (uses loaded params if not provided)
            
        Returns:
            bool: True if successful
        """
        output_path = output_path or self.par_file_path
        params = params or self.par_file_params
        
        if not params:
            print("Error: No parameters to save")
            return False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # If original file exists, use it as template to preserve format and comments
        if os.path.exists(self.par_file_path):
            with open(self.par_file_path, 'r') as f:
                content = f.read()
            
            # Update each parameter in the content
            for param_name, param_value in params.items():
                # Format the value correctly
                if isinstance(param_value, bool):
                    param_value = '.true.' if param_value else '.false.'
                
                # Replace parameter in content
                pattern = fr'{re.escape(param_name)}\s*=\s*[^#\n]*'
                replacement = f'{param_name} = {param_value}'
                content = re.sub(pattern, replacement, content)
            
            # Write updated content
            with open(output_path, 'w') as f:
                f.write(content)
        else:
            # Create new file with basic formatting
            with open(output_path, 'w') as f:
                f.write("# SPECFEM3D Par_file created by ParamManager\n")
                f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for param_name, param_value in sorted(params.items()):
                    if isinstance(param_value, bool):
                        param_value = '.true.' if param_value else '.false.'
                    f.write(f"{param_name} = {param_value}\n")
        
        print(f"Saved Par_file to {output_path}")
        return True
    
    def save_mesh_par_file(self, output_path=None, params=None):
        """
        Save parameters to Mesh_Par_file.
        
        Args:
            output_path (str, optional): Override default Mesh_Par_file path
            params (dict, optional): Parameters to save (uses loaded params if not provided)
            
        Returns:
            bool: True if successful
        """
        output_path = output_path or self.mesh_par_file_path
        params = params or self.mesh_par_file_params
        
        if not params:
            print("Error: No mesh parameters to save")
            return False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Similar approach as save_par_file
        if os.path.exists(self.mesh_par_file_path):
            with open(self.mesh_par_file_path, 'r') as f:
                content = f.read()
            
            for param_name, param_value in params.items():
                if isinstance(param_value, bool):
                    param_value = '.true.' if param_value else '.false.'
                
                pattern = fr'{re.escape(param_name)}\s*=\s*[^#\n]*'
                replacement = f'{param_name} = {param_value}'
                content = re.sub(pattern, replacement, content)
            
            with open(output_path, 'w') as f:
                f.write(content)
        else:
            with open(output_path, 'w') as f:
                f.write("# SPECFEM3D Mesh_Par_file created by ParamManager\n")
                f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for param_name, param_value in sorted(params.items()):
                    if isinstance(param_value, bool):
                        param_value = '.true.' if param_value else '.false.'
                    f.write(f"{param_name} = {param_value}\n")
        
        print(f"Saved Mesh_Par_file to {output_path}")
        return True
    
    def save_source_file(self, output_path=None, params=None):
        """
        Save parameters to CMTSOLUTION source file.
        
        Args:
            output_path (str, optional): Override default source file path
            params (dict, optional): Parameters to save (uses loaded params if not provided)
            
        Returns:
            bool: True if successful
        """
        output_path = output_path or self.source_file_path
        params = params or self.source_params
        
        if not params:
            print("Error: No source parameters to save")
            return False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get event name for header
        event_name = params.get('event_name', 'PDE 2000 01 01 00 00 00.00 0.000 0.000 0.0 0.0 0 SYNTHETIC')
        
        # Standard order of parameters for CMTSOLUTION file
        param_order = [
            'time shift', 'half duration', 'latitude', 'longitude', 'depth',
            'Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp'
        ]
        
        with open(output_path, 'w') as f:
            f.write(f"{event_name}\n")
            
            for param in param_order:
                if param in params:
                    value = params[param]
                    f.write(f"{param:<20}: {value}\n")
                else:
                    # Use default values if parameter is missing
                    if param == 'time shift':
                        f.write(f"{param:<20}: 0.0\n")
                    elif param == 'half duration':
                        f.write(f"{param:<20}: 0.0\n")
                    elif param in ['latitude', 'longitude']:
                        f.write(f"{param:<20}: 0.0\n")
                    elif param == 'depth':
                        f.write(f"{param:<20}: 10.0\n")
                    elif param.startswith('M'):
                        f.write(f"{param:<20}: 0.0\n")
        
        print(f"Saved source file to {output_path}")
        return True
    
    def save_stations_file(self, output_path=None, stations=None):
        """
        Save station information to STATIONS file.
        
        Args:
            output_path (str, optional): Override default stations file path
            stations (list, optional): Station data to save (uses loaded stations if not provided)
            
        Returns:
            bool: True if successful
        """
        output_path = output_path or self.stations_file_path
        stations = stations or self.stations
        
        if not stations:
            print("Error: No station data to save")
            return False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for station in stations:
                f.write(f"{station['name']} {station['network']} {station['lat']} {station['lon']} {station['elevation']} {station['burial']}\n")
        
        print(f"Saved {len(stations)} stations to {output_path}")
        return True
    
    def set_parameters(self, params_dict, file_type='par_file'):
        """
        Update multiple parameters at once.
        
        Args:
            params_dict (dict): Dictionary of parameters to update
            file_type (str): Type of parameter file ('par_file', 'mesh_par_file', 'source')
            
        Returns:
            bool: True if successful
        """
        if file_type == 'par_file':
            for param, value in params_dict.items():
                self.par_file_params[param] = value
            return True
        elif file_type == 'mesh_par_file':
            for param, value in params_dict.items():
                self.mesh_par_file_params[param] = value
            return True
        elif file_type == 'source':
            for param, value in params_dict.items():
                self.source_params[param] = value
            return True
        else:
            print(f"Error: Unknown file type '{file_type}'")
            return False
    
    def update_processor_settings(self, nproc):
        """
        Update processor settings in Par_file and Mesh_Par_file.
        
        Args:
            nproc (int): Number of processors
            
        Returns:
            tuple: (nproc, nproc_xi, nproc_eta)
        """
        # Calculate optimal NPROC_XI and NPROC_ETA values
        if nproc == 1:
            nproc_xi, nproc_eta = 1, 1
        elif nproc == 4:
            nproc_xi, nproc_eta = 2, 2  
        elif nproc == 8:
            nproc_xi, nproc_eta = 2, 4
        elif nproc == 16:
            nproc_xi, nproc_eta = 4, 4
        else:
            # Try to factorize into two factors close to each other
            import math
            nproc_xi = int(math.sqrt(nproc))
            nproc_eta = nproc // nproc_xi
            if nproc_xi * nproc_eta != nproc:
                print(f"Warning: {nproc} cannot be factorized evenly. Using {nproc_xi} x {nproc_eta} = {nproc_xi * nproc_eta}")
                nproc = nproc_xi * nproc_eta
        
        # Update Par_file
        self.par_file_params['NPROC'] = nproc
        
        # Update Mesh_Par_file
        self.mesh_par_file_params['NPROC_XI'] = nproc_xi
        self.mesh_par_file_params['NPROC_ETA'] = nproc_eta
        
        print(f"Updated processor settings: NPROC={nproc}, NPROC_XI={nproc_xi}, NPROC_ETA={nproc_eta}")
        return nproc, nproc_xi, nproc_eta
    
    def create_stations_grid(self, n_x, n_y, x_min=0, x_max=1000, y_min=0, y_max=1000, z=0, network="GE"):
        """
        Create a grid of stations.
        
        Args:
            n_x (int): Number of stations in X direction
            n_y (int): Number of stations in Y direction
            x_min (float): Minimum X coordinate
            x_max (float): Maximum X coordinate
            y_min (float): Minimum Y coordinate
            y_max (float): Maximum Y coordinate
            z (float): Z coordinate (elevation)
            network (str): Network code
            
        Returns:
            list: List of station dictionaries
        """
        stations = []
        idx = 1
        
        for x in np.linspace(x_min, x_max, n_x):
            for y in np.linspace(y_min, y_max, n_y):
                station = {
                    'name': f'ST{idx:03d}',
                    'network': network,
                    'lat': float(x),  # Using X as latitude
                    'lon': float(y),  # Using Y as longitude
                    'elevation': float(z),
                    'burial': 0.0
                }
                stations.append(station)
                idx += 1
        
        self.stations = stations
        print(f"Created {len(stations)} stations in a {n_x}x{n_y} grid")
        return stations
    
    def create_stations_line(self, n_stations, start_x, end_x, y=1000, z=0, network="GE"):
        """
        Create a line of stations along the X-axis.
        
        Args:
            n_stations (int): Number of stations
            start_x (float): Starting X coordinate
            end_x (float): Ending X coordinate
            y (float): Y coordinate for all stations
            z (float): Z coordinate (elevation) for all stations
            network (str): Network code
            
        Returns:
            list: List of station dictionaries
        """
        stations = []
        
        for i, x in enumerate(np.linspace(start_x, end_x, n_stations)):
            station = {
                'name': f'ST{i+1:03d}',
                'network': network,
                'lat': float(x),  # Using X as latitude
                'lon': float(y),  # Using Y as longitude
                'elevation': float(z),
                'burial': 0.0
            }
            stations.append(station)
        
        self.stations = stations
        print(f"Created {len(stations)} stations in a line from X={start_x} to X={end_x}")
        return stations
    
    def load_parameter_set(self, json_file):
        """
        Load a complete parameter set from a JSON file.
        
        Args:
            json_file (str): Path to JSON file containing parameters
            
        Returns:
            bool: True if successful
        """
        if not os.path.exists(json_file):
            print(f"Error: Parameter file {json_file} not found")
            return False
        
        try:
            with open(json_file, 'r') as f:
                params = json.load(f)
            
            if 'par_file' in params:
                self.par_file_params = params['par_file']
            
            if 'mesh_par_file' in params:
                self.mesh_par_file_params = params['mesh_par_file']
            
            if 'source' in params:
                self.source_params = params['source']
            
            if 'stations' in params:
                self.stations = params['stations']
            
            print(f"Loaded parameter set from {json_file}")
            return True
        except Exception as e:
            print(f"Error loading parameter set: {str(e)}")
            return False
    
    def save_parameter_set(self, json_file):
        """
        Save the complete parameter set to a JSON file.
        
        Args:
            json_file (str): Path to save JSON file
            
        Returns:
            bool: True if successful
        """
        params = {
            'par_file': self.par_file_params,
            'mesh_par_file': self.mesh_par_file_params,
            'source': self.source_params,
            'stations': self.stations
        }
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(json_file)), exist_ok=True)
            
            with open(json_file, 'w') as f:
                json.dump(params, f, indent=2)
            
            print(f"Saved parameter set to {json_file}")
            return True
        except Exception as e:
            print(f"Error saving parameter set: {str(e)}")
            return False
    
    def _is_float(self, value):
        """
        Check if a string can be converted to a float.
        
        Args:
            value (str): String to check
            
        Returns:
            bool: True if convertible to float
        """
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def update_parameters(self, params):
        """
        Update parameters from a dictionary loaded from a JSON file.
        
        Args:
            params (dict): Dictionary containing parameters to update
            
        Returns:
            bool: True if successful
        """
        try:
            # Update Par_file parameters
            if 'par_file' in params:
                for param, value in params['par_file'].items():
                    self.par_file_params[param] = value
                print(f"Updated {len(params['par_file'])} Par_file parameters")
            
            # Update Mesh_Par_file parameters
            if 'mesh_par_file' in params:
                for param, value in params['mesh_par_file'].items():
                    self.mesh_par_file_params[param] = value
                print(f"Updated {len(params['mesh_par_file'])} Mesh_Par_file parameters")
            
            # Update source parameters
            if 'source' in params:
                for param, value in params['source'].items():
                    self.source_params[param] = value
                print(f"Updated {len(params['source'])} source parameters")
            
            # Update stations
            if 'stations' in params:
                self.stations = params['stations']
                print(f"Updated stations list with {len(params['stations'])} stations")
            
            # Update velocity model if present
            if 'model' in params and 'layers' in params['model']:
                print(f"Updated velocity model with {len(params['model']['layers'])} layers")
                
                # Here you would apply the velocity model changes
                # For example, you might save the velocity model to a separate file
                # or update the corresponding parameters in the Par_file
                
                # Extract layer data
                layers = params['model']['layers']
                for i, layer in enumerate(layers):
                    print(f"Layer {i+1}: {layer['name']}, Vp={layer['vp']}, Vs={layer['vs']}, rho={layer['rho']}")
            
            return True
        except Exception as e:
            print(f"Error updating parameters: {str(e)}")
            return False

def main():
    """Parse command line arguments and perform requested actions."""
    parser = argparse.ArgumentParser(description="SPECFEM3D Parameter Manager")
    
    # Path arguments
    parser.add_argument("--specfem-dir", type=str, default="~/specfem3d",
                      help="Path to SPECFEM3D installation directory")
    parser.add_argument("--data-dir", type=str, 
                      help="Path to data directory containing parameter files")
    parser.add_argument("--output-dir", type=str, 
                      help="Path to output directory for parameter files")
    
    # Action arguments
    parser.add_argument("--load", action="store_true",
                      help="Load and display current parameters")
    parser.add_argument("--save", action="store_true",
                      help="Save parameters to files")
    parser.add_argument("--load-set", type=str, 
                      help="Load parameter set from JSON file")
    parser.add_argument("--save-set", type=str, 
                      help="Save parameter set to JSON file")
    
    # Parameter update arguments
    parser.add_argument("--processors", type=int, 
                      help="Set number of processors (updates Par_file and Mesh_Par_file)")
    parser.add_argument("--create-stations", type=str, 
                      help="Create stations (format: NxM for grid, N for line)")
    parser.add_argument("--update-par", type=str, nargs="+",
                      help="Update Par_file parameters (format: PARAM=VALUE)")
    parser.add_argument("--update-mesh", type=str, nargs="+",
                      help="Update Mesh_Par_file parameters (format: PARAM=VALUE)")
    parser.add_argument("--update-source", type=str, nargs="+",
                      help="Update source parameters (format: PARAM=VALUE)")
    
    args = parser.parse_args()
    
    # Initialize parameter manager
    pm = ParamManager(args.specfem_dir, args.data_dir)
    
    # Load parameter set if requested
    if args.load_set:
        pm.load_parameter_set(args.load_set)
    
    # Update parameters if requested
    if args.processors:
        pm.update_processor_settings(args.processors)
    
    if args.create_stations:
        if 'x' in args.create_stations.lower():
            # Grid format: NxM
            parts = args.create_stations.lower().split('x')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                n_x, n_y = int(parts[0]), int(parts[1])
                pm.create_stations_grid(n_x, n_y)
            else:
                print("Error: Invalid station grid format. Use NxM (e.g., 5x4)")
        elif args.create_stations.isdigit():
            # Line format: N
            n_stations = int(args.create_stations)
            pm.create_stations_line(n_stations, 500, 1500)
        else:
            print("Error: Invalid station format. Use NxM for grid or N for line")
    
    if args.update_par:
        params = {}
        for param_str in args.update_par:
            if '=' in param_str:
                name, value = param_str.split('=', 1)
                # Convert value to appropriate type
                if value.isdigit():
                    value = int(value)
                elif pm._is_float(value):
                    value = float(value)
                elif value.lower() in ['.true.', 'true']:
                    value = True
                elif value.lower() in ['.false.', 'false']:
                    value = False
                params[name] = value
        pm.set_parameters(params, 'par_file')
    
    if args.update_mesh:
        params = {}
        for param_str in args.update_mesh:
            if '=' in param_str:
                name, value = param_str.split('=', 1)
                # Convert value to appropriate type
                if value.isdigit():
                    value = int(value)
                elif pm._is_float(value):
                    value = float(value)
                elif value.lower() in ['.true.', 'true']:
                    value = True
                elif value.lower() in ['.false.', 'false']:
                    value = False
                params[name] = value
        pm.set_parameters(params, 'mesh_par_file')
    
    if args.update_source:
        params = {}
        for param_str in args.update_source:
            if '=' in param_str:
                name, value = param_str.split('=', 1)
                # Convert value to appropriate type
                if pm._is_float(value):
                    value = float(value)
                params[name] = value
        pm.set_parameters(params, 'source')
    
    # Display loaded parameters if requested
    if args.load:
        print("\n=== Par_file Parameters ===")
        for param, value in sorted(pm.par_file_params.items()):
            print(f"{param} = {value}")
        
        print("\n=== Mesh_Par_file Parameters ===")
        for param, value in sorted(pm.mesh_par_file_params.items()):
            print(f"{param} = {value}")
        
        print("\n=== Source Parameters ===")
        for param, value in sorted(pm.source_params.items()):
            print(f"{param}: {value}")
        
        print(f"\n=== Stations ({len(pm.stations)}) ===")
        for i, station in enumerate(pm.stations[:5]):
            print(f"{station['name']} {station['network']} {station['lat']} {station['lon']} {station['elevation']} {station['burial']}")
        if len(pm.stations) > 5:
            print(f"... {len(pm.stations) - 5} more stations ...")
    
    # Save parameters if requested
    if args.save:
        # Set output directory if specified
        if args.output_dir:
            output_dir = os.path.expanduser(args.output_dir)
            par_file_path = os.path.join(output_dir, "Par_file")
            mesh_par_file_path = os.path.join(output_dir, "meshfem3D_files", "Mesh_Par_file")
            source_file_path = os.path.join(output_dir, "CMTSOLUTION")
            stations_file_path = os.path.join(output_dir, "STATIONS")
        else:
            par_file_path = None
            mesh_par_file_path = None
            source_file_path = None
            stations_file_path = None
        
        # Save files
        pm.save_par_file(par_file_path)
        pm.save_mesh_par_file(mesh_par_file_path)
        pm.save_source_file(source_file_path)
        pm.save_stations_file(stations_file_path)
    
    # Save parameter set if requested
    if args.save_set:
        pm.save_parameter_set(args.save_set)

if __name__ == "__main__":
    main() 