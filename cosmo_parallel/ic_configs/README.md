MUSIC config for the `music25` matched realization (seed[9]=45678, 25 Mpc/h, h=0.71, Om=0.27, Ob=0.045).
Source of truth lives in Nyx/Exec/LyA/music25_ics/ (nested Nyx checkout, not tracked here).
Identical seeds were used for music25_matched_n512.conf, so the "true" and "matched" IC families are the same realization:
- music25_n512_true_generic.hdf5 -> resample_music_generic.py -> n400_true (DiffHydro runs)
- music25_matched_n512.nyx / matched_n256.nyx (Nyx baseline runs n512_gpu, n256_gpu)
