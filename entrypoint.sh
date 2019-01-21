echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
pip uninstall --yes Pillow
yes | pip install Pillow==4.3
bash