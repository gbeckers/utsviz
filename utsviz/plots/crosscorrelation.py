import numpy as np
from uts import ismcuts


def mc_crosscorrelation_images(cc):
    if not ismcuts(cc):
        raise TypeError("cc should be a multichannel uniform time series")
    nchannels = np.sqrt(cc.nchannels)
    if not (nchannels % 2.0) == 0:
        raise TypeError("cc should have a number of channels that is a power of 2")
    nchannels = int(nchannels)
    m = cc.samples.get().reshape(nchannels, nchannels, -1)
    # This function was never finished: it computes m and then stops. Made
    # explicit rather than silently returning None.
    raise NotImplementedError(
        "mc_crosscorrelation_images has never been implemented beyond this "
        "point")
