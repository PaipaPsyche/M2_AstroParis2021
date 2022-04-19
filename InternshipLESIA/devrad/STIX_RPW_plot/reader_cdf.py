# -*- coding: utf-8 -*-
def read_hfr_autoch_full(filepath, sensor=9, start_index=0, end_index=-99):
    import os
    os.environ["CDF_LIB"] = "/home/localuser/Documents/CDF/src/lib"
    from spacepy import pycdf
    import numpy as np
    import datetime

    with pycdf.CDF ( filepath ) as l2_cdf_file:

        frequency = l2_cdf_file[ 'FREQUENCY' ][ : ]  # / 1000.0  # frequency in MHz
        nn = np.size ( l2_cdf_file[ 'Epoch' ][ : ] )
        if end_index == -99:
            end_index = nn
        frequency = frequency[ start_index:end_index ]
        epochdata = l2_cdf_file[ 'Epoch' ][ start_index:end_index ]
        sensor_config = np.transpose (
            l2_cdf_file[ 'SENSOR_CONFIG' ][ start_index:end_index, : ]
        )
        agc1_data = np.transpose ( l2_cdf_file[ 'AGC1' ][ start_index:end_index ] )
        agc2_data = np.transpose ( l2_cdf_file[ 'AGC2' ][ start_index:end_index ] )
        sweep_num = l2_cdf_file[ 'SWEEP_NUM' ][ start_index:end_index ]
        cal_points = (
            l2_cdf_file[ 'FRONT_END' ][ start_index:end_index ] == 1
        ).nonzero ()
    frequency = frequency[ cal_points[ 0 ] ]
    epochdata = epochdata[ cal_points[ 0 ] ]
    sensor_config = sensor_config[ :, cal_points[ 0 ] ]
    agc1_data = agc1_data[ cal_points[ 0 ] ]
    agc2_data = agc2_data[ cal_points[ 0 ] ]
    sweep_numo = sweep_num[ cal_points[ 0 ] ]
    ssweep_num = sweep_numo
    timet = epochdata

    # deltasw = sweep_numo[ 1:: ] - sweep_numo[ 0:np.size ( sweep_numo ) - 1 ]
    deltasw = abs ( np.double ( sweep_numo[ 1:: ] ) - np.double ( sweep_numo[ 0:np.size ( sweep_numo ) - 1 ] ) )
    xdeltasw = np.where ( deltasw > 100 )
    xdsw = np.size ( xdeltasw )
    if xdsw > 0:
        xdeltasw = np.append ( xdeltasw, np.size ( sweep_numo ) - 1 )
        nxdeltasw = np.size ( xdeltasw )
        for inswn in range ( 0, nxdeltasw - 1 ):
            # sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] = sweep_num[
            # xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] ] + \
            # sweep_numo[ xdeltasw[ inswn ] ]
            sweep_num[ xdeltasw[ inswn ] + 1:xdeltasw[ inswn + 1 ] + 1 ] = sweep_num[
                                                                           xdeltasw[ inswn ] + 1:xdeltasw[
                                                                                                     inswn + 1 ] + 1 ] + \
                                                                           sweep_numo[ xdeltasw[ inswn ] ]
    sens0 = (sensor_config[ 0, : ] == sensor).nonzero ()[ 0 ]
    sens1 = (sensor_config[ 1, : ] == sensor).nonzero ()[ 0 ]
    print("  sensors: ",np.shape(sens0),np.shape(sens1))
    psens0 = np.size ( sens0 )
    psens1 = np.size ( sens1 )

    if (np.size ( sens0 ) > 0 and np.size ( sens1 ) > 0):
        agc = np.append ( np.squeeze ( agc1_data[ sens0 ] ), np.squeeze ( agc2_data[ sens1 ] ) )
        frequency = np.append ( np.squeeze ( frequency[ sens0 ] ), np.squeeze ( frequency[ sens1 ] ) )
        sens = np.append ( sens0, sens1 )
        timet_ici = np.append ( timet[ sens0 ], timet[ sens1 ] )
    else:
        if (np.size ( sens0 ) > 0):
            agc = np.squeeze ( agc1_data[ sens0 ] )
            frequency = frequency[ sens0 ]
            sens = sens0
            timet_ici = timet[ sens0 ]
        if (np.size ( sens1 ) > 0):
            agc = np.squeeze ( agc2_data[ sens1 ] )
            frequency = frequency[ sens1 ]
            sens = sens1
            timet_ici = timet[ sens1 ]
        if (np.size ( sens0 ) == 0 and np.size ( sens1 ) == 0):
            print('  no data at all ?!?')
            V = (321)
            V = np.zeros ( V ) + 1.0
            time = np.zeros ( 128 )
            sweepn_HFR = 0.0
    #           return {
    #               'voltage': V,
    #               'time': time,
    #               'frequency': frequency,
    #               'sweep': sweepn_HFR,
    #               'sensor': sensor,
    #           }
    ord_time = np.argsort ( timet_ici )
    timerr = timet_ici[ ord_time ]
    sens = sens[ ord_time ]
    agc = agc[ ord_time ]
    frequency = frequency[ ord_time ]
    maxsweep = max ( sweep_num[ sens ] )
    minsweep = min ( sweep_num[ sens ] )
    sweep_num = sweep_num[ sens ]

    V1 = np.zeros ( 321 ) - 99.
    V = np.zeros ( 321 )
    freq_hfr1 = np.zeros ( 321 ) - 99.
    freq_hfr = np.zeros ( 321 )
    time = 0.0
    sweepn_HFR = 0.0
    # ind_freq = [(frequency - 0.375) / 0.05]
    ind_freq = [ (frequency - 375.) / 50. ]
    ind_freq = np.squeeze ( ind_freq )
    ind_freq = ind_freq.astype ( int )
    for ind_sweep in range ( minsweep, maxsweep + 1 ):
        ppunt = (sweep_num == ind_sweep).nonzero ()[ 0 ]
        xm = np.size ( ppunt )
        if xm > 0:
            V1[ ind_freq[ ppunt ] ] = agc[ ppunt ]
            freq_hfr1[ ind_freq[ ppunt ] ] = frequency[ ppunt ]
            # print(frequency[ppunt])
        if np.max ( V1 ) > 0.0:
            V = np.vstack ( (V, V1) )
            freq_hfr = np.vstack ( (freq_hfr, freq_hfr1) )
            sweepn_HFR = np.append ( sweepn_HFR, sweep_num[ ppunt[ 0 ] ] )
        V1 = np.zeros ( 321 ) - 99
        freq_hfr1 = np.zeros ( 321 )  # - 99
        if xm > 0:
            time = np.append ( time, timerr[ min ( ppunt ) ] )
    # sys.exit ( "sono qui" )
    V = np.transpose ( V[ 1::, : ] )
    time = time[ 1:: ]
    sweepn_HFR = sweepn_HFR[ 1:: ]
    freq_hfr = np.transpose ( freq_hfr[ 1::, : ] )
    return {
        'voltage': V,
        'time': time,
        'frequency': freq_hfr,
        'sweep': sweepn_HFR,
        'sensor': sensor,
    }
