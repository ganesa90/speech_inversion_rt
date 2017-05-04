# speech_inversion_rt
Real time version of a speech inversion system

Run the code speech_inversion_rt.py
The code reads audio from a micophone connected to your computer and plots the waveform, along with 3 tract variables in real time
The 3 tract variables are Lip Aperture, Tongue Body Constriction Degree, and Tongue Tip Constriction Degree

# Important Notes
1. Do the experiment in a quiet environment
2. Immediately after starting the code, read 4-5 sentences for around 30 seconds before you start experimenting with it. The model needs to calibrate to your voice before it starts giving correct and stable outputs.
3. Lesser the silence intervals you leave is better. Currently this code does not contain a speech activity detector. prolonged silence for a long time will make the system go out of calibration. It is better to restart the code and recalibrate.
4. The speech inversion system was trained on 36 speakers from the X-ray microbeam (XRMB) dataset converted to tract Variables. The accuracy on the XRMB test set (5 unseen speakers) is 78% across all TVs (correlation with actual TVs)

