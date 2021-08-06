source env/.venv/bin/activate

export DISPLAY=`grep -oP "(?<=nameserver ).+" /etc/resolv.conf`:0.0

python3 src/image_process_v2.2.py