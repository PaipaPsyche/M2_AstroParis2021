The file VDF_4_Masterclass.txt" contains a typical example of an electron
Velocity Distribution Function (VDF) observed in the Solar Wind

This VDF is an array of dimensions [15, 88], corresponding to 15 energy channels
and 88 angular bins in the sky. Tha data are stored in the files as 15*88=1320 lines
containing the following parameters

E(eV),dE(eV), Phi(deg), dPhi(deg), Theta(deg), dTheta(deg), Log10(FDV) with FDV in s^3/km^3/cm^3
- where E and dE are respectively the energy and energy widths of the energy channels
- Phi and Theta are the azimuths and elevations of the angular bin of observations
- dPhi and dTheta their angular widths
- Log10(FDV) is the Logarithm of FDV which is expressed in s^3/km^3/cm^3

Tasks to be done before the lecture :
1) read the ascii file and built the array fdv[15, 88] & E([15, 88]
2) plot on tha same figure and for all the 88 angular bins fdv(E)
3) Guess what is the typical temperature of the solar wind electrons



