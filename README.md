## To set up a virtual environment for cacdi_attention_model:

Identify a folder for the virtual environment files (I'll use /envpath below)
Identify a folder for development (I'll use /devpath)

1. Create python virtual environment

   virtualenv --system-site-packages /envpath/han

2. Activate virtual environment

   source /envpath/han/bin/activate

3. Check out cacdi_attention_model repo

   cd /devpath
   git clone git@git.labs.nuance.com:xiaohua.liu/cacdi_attention_model.git

4. Install cacdi_attention_model in virtual environment

   pip install --no-index -f /hcnlp/projects/cacdi/tools/python/wheelhouse -e /devpath/cacdi_attention_model/ --process-dependency-links 

5. Create environment variables  

   export HDF5_DIR=/hcnlp/projects/cacdi/tools/lib/hdf5/hdf5-1.8.16
   export HDF5_VERSION=1.8.16
   export CACDI_DATA=/hcnlpdata/cacdi
   export GPU_LAUNCHER=/hcnlp/projects/cacdi/release/common_utils/latest/common_utils/process/gpu_nrg8_process_launcher.sh

6. Verify installation is correct

    python -m unittest discover -s ./test/