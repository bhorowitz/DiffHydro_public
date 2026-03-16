import numpy as np
from scipy.stats import norm

__all__ = ['power_spectrum']

#pure numpy


def _initialize_pk(shape, boxsize, kmin, dk):
    """
       Helper function to initialize various (fixed) values for powerspectra... not differentiable!
    """
    I = np.eye(len(shape), dtype='int') * -2 + 1

    W = np.empty(shape, dtype='f4')
    W[...] = 2.0
    W[..., 0] = 1.0
    W[..., -1] = 1.0

    kmax = np.pi * np.min(np.array(shape)) / np.max(np.array(boxsize)) + dk / 2
    kedges = np.arange(kmin, kmax, dk)

    k = [
        np.fft.fftfreq(N, 1. / (N * 2 * np.pi / L))[:pkshape].reshape(kshape)
        for N, L, kshape, pkshape in zip(shape, boxsize, I, shape)
    ]
    kmag = sum(ki**2 for ki in k)**0.5

    xsum = np.zeros(len(kedges) + 1)
    Nsum = np.zeros(len(kedges) + 1)

    dig = np.digitize(kmag.flat, kedges)

    xsum.flat += np.bincount(dig, weights=(W * kmag).flat, minlength=xsum.size)
    Nsum.flat += np.bincount(dig, weights=W.flat, minlength=xsum.size)
    return dig, Nsum, xsum, W, k, kedges


def power_spectrum(field, kmin=5, dk=0.5, boxsize=False):
    """
    Calculate the powerspectra given real space field

    Args:

        field: real valued field
        kmin: minimum k-value for binned powerspectra
        dk: differential in each kbin
        boxsize: length of each boxlength (can be strangly shaped?)

    Returns:

        kbins: the central value of the bins for plotting
        power: real valued array of power in each bin

  """
    shape = field.shape

    #initialze values related to powerspectra (mode bins and weights)
    dig, Nsum, xsum, W, k, kedges = _initialize_pk(shape, boxsize, kmin, dk)

    #fast fourier transform
    fft_image = np.fft.fftn(field)

    #absolute value of fast fourier transform
    pk = np.real(fft_image * np.conj(fft_image))

    #calculating powerspectra
    real = np.real(pk).reshape([-1])
    imag = np.imag(pk).reshape([-1])

    Psum = np.bincount(dig, weights=(W.flatten() * imag),
                        minlength=xsum.size) * 1j
    Psum += np.bincount(dig, weights=(W.flatten() * real), minlength=xsum.size)

    P = ((Psum / Nsum)[1:-1] * boxsize.prod()).astype('float32')

    #normalization for powerspectra
    norm = np.prod(np.array(shape[:])).astype('float32')**2

    #find central values of each bin
    kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2

    return kbins, P / norm

def cross_correlation(field_a,field_b, kmin=5, dk=0.5, boxsize=False):
    k,pab = cross_correlation_coefficients(field_a,field_b, kmin=kmin, dk=dk, boxsize=boxsize)

    _,pa = power_spectrum(field_a,kmin=kmin, dk=dk, boxsize=boxsize)
    _,pb = power_spectrum(field_b,kmin=kmin, dk=dk, boxsize=boxsize)

    return k, pab/np.sqrt(pa*pb)

def cross_correlation_coefficients(field_a,field_b, kmin=5, dk=0.5, boxsize=False):
  """
    Calculate the cross correlation coefficients given two real space field
    
    Args:
        
        field_a: real valued field 
        field_b: real valued field 
        kmin: minimum k-value for binned powerspectra
        dk: differential in each kbin
        boxsize: length of each boxlength (can be strangly shaped?)
    
    Returns:
        
        kbins: the central value of the bins for plotting
        P / norm: normalized cross correlation coefficient between two field a and b 
        
  """
  shape = field_a.shape

  #initialze values related to powerspectra (mode bins and weights)
  dig, Nsum, xsum, W, k, kedges = _initialize_pk(shape, boxsize, kmin, dk)

  #fast fourier transform
  fft_image_a = np.fft.fftn(field_a)
  fft_image_b = np.fft.fftn(field_b)

  #absolute value of fast fourier transform
  pk = fft_image_a * np.conj(fft_image_b)

  #calculating powerspectra
  real = np.real(pk).reshape([-1])
  imag = np.imag(pk).reshape([-1])

  Psum = np.bincount(dig, weights=(W.flatten() * imag), minlength=xsum.size) * 1j
  Psum += np.bincount(dig, weights=(W.flatten() * real), minlength=xsum.size)

  P = ((Psum / Nsum)[1:-1] * boxsize.prod()).astype('float32')

  #normalization for powerspectra
  norm = np.prod(np.array(shape[:])).astype('float32')**2

  #find central values of each bin
  kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2

  return kbins, P / norm


def gaussian_smoothing(im, sigma):
    """
  im: 2d image
  sigma: smoothing scale in px
  """
    # Compute k vector
    kvec = np.stack(np.meshgrid(np.fft.fftfreq(im.shape[0]),
                                  np.fft.fftfreq(im.shape[1])),
                     axis=-1)
    k = np.linalg.norm(kvec, axis=-1)
    # We compute the value of the filter at frequency k
    filter = norm.pdf(k, 0, 1. / (2. * np.pi * sigma))
    filter /= filter[0, 0]

    return np.fft.ifft2(np.fft.fft2(im) * filter).real