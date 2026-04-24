#!/bin/bash
set -e

echo "Installing docking dependencies via uv..."

uv add rdkit
uv add meeko
uv add vina
uv add biopython
uv add pandas

echo "Dependencies installed successfully!"
