# IR-multichannel
Calculate multichannel IR files through deconvolution of a recorded sweep with a Jupyter notebook.

If you do not have Jupyter, install it from https://jupyter.org/install

Download and copy all code to a folder on your computer. 

Run Jupyter from a terminal via "jupyter notebook", open the Jupyter notebook example and execute the cells from top to bottom. 

Try to run the examples using mono, stereo and 4 channels wave files of recorded sweeps. Some examples The folder also contains the original sweep file (mono). 

This code generates IR files with the same number of channels as the recorded sweep you are using.

You can use the resulting .wav file with plugins like MatrixConv (https://leomccormack.github.io/sparta-site/docs/plugins/sparta-suite/#matrixconv) in combination with HO-SIRR (https://leomccormack.github.io/sparta-site/docs/plugins/hosirr/) for adding an ambisonics room to your track in real time. 
