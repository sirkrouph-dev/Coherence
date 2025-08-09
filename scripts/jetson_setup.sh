#!/bin/bash
# Jetson Nano Neuromorphic System Deployment Script

echo "Setting up Jetson Nano for neuromorphic computing..."

# Update system
sudo apt update
sudo apt upgrade -y

# Install Python dependencies
sudo apt install -y python3-pip python3-dev
pip3 install numpy matplotlib scipy scikit-learn pandas

# Install Jetson-specific packages
sudo apt install -y python3-opencv
pip3 install cupy-cuda11x  # For CUDA acceleration

# Install neuromorphic system
cd /home/nano/neuron
pip3 install -r requirements.txt

# Set up performance monitoring
sudo apt install -y lm-sensors
sudo sensors-detect --auto

# Create startup script
cat > /home/nano/start_neuromorphic.sh << 'EOF'
#!/bin/bash
cd /home/nano/neuron
python3 demo/jetson_demo.py
EOF

chmod +x /home/nano/start_neuromorphic.sh

echo "Jetson Nano setup complete!"
echo "Run: ./start_neuromorphic.sh"
