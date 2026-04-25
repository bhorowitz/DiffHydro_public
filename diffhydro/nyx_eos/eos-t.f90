! Tabulates cooling and UV heating rates.
!
!     Units are CGS (and temperature in K).
!
!     Two sets of rates, as in:
!      1) Katz, Weinberg & Hernquist, 1996: Astrophysical Journal Supplement
!         v.105, p.19
!      2) Lukic et al.
!
!     NOTE: This is executed only once per run, and rates are ugly, thus
!           execution efficiency is not important, but readability of
!           the code is. -- Zarija
!

module atomic_rates
  implicit none

  public :: tabulate_rates, interp_to_this_z

  ! Photo-rates (from file)
  integer, parameter, public :: NCOOLFILE=301 !59 !214
  double precision, dimension(NCOOLFILE), public :: lzr
  double precision, dimension(NCOOLFILE), public :: rggh0, rgghe0, rgghep
  double precision, dimension(NCOOLFILE), public :: reh0, rehe0, rehep

  ! Other rates (from equations)
  integer, parameter, public :: NCOOLTAB=2000
  double precision, dimension(NCOOLTAB+1), public :: AlphaHp, AlphaHep, AlphaHepp, Alphad
  double precision, dimension(NCOOLTAB+1), public :: GammaeH0, GammaeHe0, GammaeHep
  double precision, dimension(NCOOLTAB+1), public :: BetaH0, BetaHe0, BetaHep, Betaff1, Betaff4
  double precision, dimension(NCOOLTAB+1), public :: RecHp, RecHep, RecHepp

  double precision, public, save :: this_z, ggh0, gghe0, gghep, eh0, ehe0, ehep

  double precision, parameter, public :: TCOOLMIN = 0.0d0, TCOOLMAX = 9.0d0  ! in log10
  double precision, parameter, public :: MPROTON = 1.6726231d-24, BOLTZMANN = 1.38064e-16
  double precision, public :: XHYDROGEN = 0.76d0
  double precision, public :: YHELIUM = 7.8947368421d-2  ! (1-X)/(4*X)

  ! EOS gamma
  double precision, parameter :: gamma_const   = 5.0d0 / 3.0d0
  double precision, parameter :: gamma_minus_1 = gamma_const - 1.0d0

  contains

    subroutine tabulate_rates()
      implicit none
      logical, save :: first=.true.
      logical, parameter :: use_katz96=.false.
      double precision, parameter :: t3=1.0d3, t5=1.0d5, t6=1.0d6
      double precision, parameter :: uvb_rescale = 1.0d0
      integer :: i
      double precision :: A, E, U, X, aa, b, m, y, T0, T1
      double precision :: t, sqrt_t, corr_term, deltaT

      !$OMP CRITICAL(TREECOOL_READ)
      if (first) then

        first = .false.

        ! Read in photoionization rates and heating from a file
        !open(unit=11,file='TREECOOL_fg_dec11',status='old')
        !open(unit=11,file='TREECOOL_hm_12',status='old')
        !open(unit=11,file='TREECOOL_zl_dec14',status='old')
        open(unit=11,file='TREECOOL_middle',status='old')
        do i = 1, NCOOLFILE
          read(11,*) lzr(i), rggh0(i), rgghe0(i), rgghep(i), &
                             reh0(i),  rehe0(i),  rehep(i)
          rggh0(i)  = uvb_rescale * rggh0(i)
          reh0(i)   = uvb_rescale * reh0(i)
          rgghe0(i) = uvb_rescale * rgghe0(i)
          rehe0(i)  = uvb_rescale * rehe0(i)
          rgghep(i) = uvb_rescale * rgghep(i)
          rehep(i)  = uvb_rescale * rehep(i)
        end do
        close(11)

        ! Initialize cooling tables
        t = 10.0d0**TCOOLMIN
        deltaT = (TCOOLMAX - TCOOLMIN)/NCOOLTAB

        if (use_katz96) then
          ! Using rates are as in Katz et al. 1996
          do i = 1, NCOOLTAB+1

            sqrt_t = dsqrt(t)
            corr_term    = 1.d0 / (1.0d0 + sqrt_t/dsqrt(t5))

            ! Recombination rates
            ! Dielectronic recombination rate of singly ioniozed helium
            Alphad(i)    = 1.90d-03/(t*sqrt_t) * dexp(-4.7d5/t) * (1.0d0+0.3d0*dexp(-9.4d4/t))
            AlphaHp(i)   = 8.40d-11/sqrt_t * (t/t3)**(-0.2d0) / (1.0d0 + (t/t6)**0.7d0)
            AlphaHep(i)  = 1.50d-10 * t**(-0.6353d0)
            AlphaHepp(i) = 3.36d-10/sqrt_t * (t/t3)**(-0.2d0) / (1.0d0 + (t/t6)**0.7d0)

            ! Collisional ionization rates
            GammaeH0(i)  = 5.85d-11*sqrt_t * dexp(-157809.1d0/t) * corr_term
            GammaeHe0(i) = 2.38d-11*sqrt_t * dexp(-285335.4d0/t) * corr_term
            GammaeHep(i) = 5.68d-12*sqrt_t * dexp(-631515.0d0/t) * corr_term

            ! Collisional ionization & excitation cooling rates
            BetaH0(i)  = 7.5d-19 * dexp(-118348.0d0/t) * corr_term + 2.171d-11*GammaeH0(i)
            BetaHe0(i) = 3.941d-11 * GammaeHe0(i)
            BetaHep(i) = 5.54d-17 * t**(-0.397d0) * dexp(-473638.0d0/t) * corr_term &
                         + 8.715d-11 * GammaeHep(i)

            ! Recombination cooling rates
            RecHp(i)   = 1.036d-16 * t * AlphaHp(i)
            RecHep(i)  = 1.036d-16 * t * AlphaHep(i) + 6.526d-11 * Alphad(i)
            RecHepp(i) = 1.036d-16 * t * AlphaHepp(i)

            ! Free-free cooling rate
            Betaff1(i) = 1.42d-27 * sqrt_t * (1.1d0 + 0.34d0*dexp(-(5.5d0 - dlog10(t))**2 / 3.0d0))
            Betaff4(i) = Betaff1(i)

            t = t*10.0d0**deltaT
          enddo
        else
          ! Using rates are as in Lukic et al.
          do i = 1, NCOOLTAB+1

            sqrt_t = dsqrt(t)

            !
            ! Recombination rates section
            !

            ! Dielectronic recombination rate of singly ionized helium.
            ! Aldrovandi and Pequignot 1973.
            Alphad(i) = 1.9d-03 / (t*sqrt_t) * dexp(-4.7d5 / t) &
                        * (1.0d0 + 0.3d0*dexp(-9.4d4 / t))

            ! Ionized hydrogen.
            ! Verner and Ferland 1996.
            aa = 7.982d-11
            b = 0.7480d0
            T0 = 3.148d0
            T1 = 7.036d5
            AlphaHp(i) = aa / ( dsqrt(t / T0) &
                               * (1.0d0 + dsqrt(t / T0))**(1 - b) &
                               * (1.0d0 + dsqrt(t / T1))**(1 + b) )

            ! Singly ionized helium.
            ! Verner and Ferland 1996.
            if (t .le. 1.0d6) then
              aa = 3.294d-11
              b = 0.6910d0
              T0 = 15.54d0
              T1 = 3.676d7
            else
              aa = 9.356d-10
              b = 0.7892d0
              T0 = 4.266d-2
              T1 = 4.677d6
            endif

            AlphaHep(i) = aa / ( dsqrt(t / T0) &
                                * (1.0d0 + dsqrt(t / T0))**(1 - b) &
                                * (1.0d0 + dsqrt(t / T1))**(1 + b) )

            ! Doubly ionized helium.
            ! Verner and Ferland 1996.
            aa = 1.891d-10
            b = 0.7524d0
            T0 = 9.370d0
            T1 = 2.774d6
            AlphaHepp(i) = aa / ( dsqrt(t / T0) &
                                 * (1.0d0 + dsqrt(t / T0))**(1 - b) &
                                 * (1.0d0 + dsqrt(t / T1))**(1 + b) )

            !
            ! Collisional ionization rates section
            ! Voronov 1997.
            !

            ! Neutral hydrogen.
            A = 2.91d-8
            E = 13.6d0
            X = 0.232d0
            m = 0.39d0
            U = 1.16045d4 * E / t
            GammaeH0(i) = A * U**m * dexp(-U) / (X + U)

            ! Neutral helium.
            A = 1.75d-8
            E = 24.6d0
            X = 0.180d0
            m = 0.35d0
            U = 1.16045d4 * E / t
            GammaeHe0(i) = A * U**m * dexp(-U) / (X + U)

            ! Singly ionized helium.
            A = 2.05d-9
            E = 54.4d0
            X = 0.265d0
            m = 0.25d0
            U = 1.16045d4 * E / t
            GammaeHep(i) = A * (1.0d0 + dsqrt(U)) * U**m * dexp(-U) / (X + U)

            !
            ! Collisional ionization and excitation cooling rates section
            !

            ! Neutral hydrogen.
            ! Scholz and Walters 1991.
            y = dlog(t)
            if (t .le. 1.0d5) then
              BetaH0(i) = 1.0d-20 * dexp( 2.137913d2 - 1.139492d2*y + 2.506062d1*y**2 &
                                          - 2.762755d0*y**3 + 1.515352d-1*y**4 &
                                          - 3.290382d-3*y**5 - 1.18415d5 / t )
            else
              BetaH0(i) = 1.0d-20 * dexp( 2.7125446d2 - 9.8019455d1*y + 1.400728d1*y**2 &
                                          - 9.780842d-1*y**3 + 3.356289d-2*y**4 &
                                          - 4.553323d-4*y**5 - 1.18415d5 / t )
            endif

            ! Neutral helium.
            ! Black 1981.
            corr_term = 1.0d0 / (1.0d0 + sqrt_t / dsqrt(5.0d7))
            BetaHe0(i) = 9.38d-22 * sqrt_t * dexp(-285335.4d0 / t) * corr_term

            ! Singly ionized helium.
            BetaHep(i) = ( 5.54d-17 * t**(-0.397d0) * dexp(-473638.0d0 / t) &
                           + 4.85d-22 * sqrt_t * dexp(-631515.0d0 / t) ) * corr_term

            !
            ! Recombination cooling rates section.
            ! Black 1981.
            !

            ! Ionized hydrogen
            RecHp(i)   = 2.851d-27 * sqrt_t &
                         * (5.914d0 - 0.5d0 * dlog(t) + 1.184d-2 * t**(1.0d0/3.0d0))
            ! Singly ionized helium
            RecHep(i)  = 1.55d-26 * t**0.3647 + 1.24d-13 / (t*sqrt_t) &
                         * dexp(-4.7d5 / t) * (1.0d0 + 0.3d0 * dexp(-9.4d4 / t))
            ! Doubly ionized helium
            RecHepp(i) = 1.14d-26 * sqrt_t &
                         * (6.607d0 - 0.5d0 * dlog(t) + 7.459d-3 * t**(1.0d0/3.0d0))

            !
            ! Free-free cooling rate section
            ! Shapiro and Kang 1987.
            !

            ! Z = 1 species, HII and HeII.
            if (t .le. 3.2d5) then
              Betaff1(i) = 1.426d-27 * sqrt_t * (0.79464d0 + 0.1243d0 * dlog10(t))
            else
              Betaff1(i) = 1.426d-27 * sqrt_t * (2.13164d0 - 0.1240d0 * dlog10(t))
            endif

            ! Z = 2 species, HeIII.
            if (t / 4.0d0 .le. 3.2d5) then
              Betaff4(i) = 1.426d-27 * sqrt_t * 4.0d0 * (0.79464d0 + 0.1243d0 * dlog10(t))
            else
              Betaff4(i) = 1.426d-27 * sqrt_t * 4.0d0 * (2.13164d0 - 0.1240d0 * dlog10(t))
            endif

            ! Don't forget to update temp!
            t = t*10.0d0**deltaT

          enddo
        endif ! if use katz rates

      end if  ! first_call
      !$OMP END CRITICAL(TREECOOL_READ)

    end subroutine tabulate_rates


    subroutine interp_to_this_z(z)

      double precision, intent(in) :: z
      double precision :: lopz, fact
      integer :: i, j

      this_z = z
      lopz   = dlog10(1.0d0 + z)

      if (lopz .ge. lzr(NCOOLFILE)) then
        ggh0  = 0.0d0
        gghe0 = 0.0d0
        gghep = 0.0d0
        eh0   = 0.0d0
        ehe0  = 0.0d0
        ehep  = 0.0d0
        return
      endif

      j = 1
      if (lopz .le. lzr(1)) then
        j = 1
      else
        do i = 2, NCOOLFILE
          if (lopz .lt. lzr(i)) then
            j = i-1
            exit
          endif
        enddo
      endif

      fact  = (lopz - lzr(j)) / (lzr(j+1) - lzr(j))

      ggh0  =  rggh0(j) + ( rggh0(j+1) -  rggh0(j)) * fact
      gghe0 = rgghe0(j) + (rgghe0(j+1) - rgghe0(j)) * fact
      gghep = rgghep(j) + (rgghep(j+1) - rgghep(j)) * fact
      eh0   =   reh0(j) + (  reh0(j+1) -   reh0(j)) * fact
      ehe0  =  rehe0(j) + ( rehe0(j+1) -  rehe0(j)) * fact
      ehep  =  rehep(j) + ( rehep(j+1) -  rehep(j)) * fact

    end subroutine interp_to_this_z

end module atomic_rates


module eos

  implicit none

  ! Routines:
  public  :: nyx_eos
  public :: iterate_ne, ion_n

  contains

    subroutine nyx_eos(z, rho, T, n_hi) &
         bind(c, name='c_nyx_eos')
      ! This is for skewers analysis code, input is in CGS
      use atomic_rates, ONLY: XHYDROGEN, MPROTON, &
                              tabulate_rates, interp_to_this_z

      ! In/out variables
      double precision, intent(in   ) :: z, rho, T
      double precision, intent(  out) :: n_hi

      double precision :: nh, nh0, nhp, nhe0, nhep, nhepp, ne
      logical, save :: first_call=.true.

      nh  = rho*XHYDROGEN/MPROTON
      ne  = 1.0d0 ! Guess

      if (first_call) then
        first_call = .false.
        call tabulate_rates()
        call interp_to_this_z(z)
      endif

      call iterate_ne(z, T, nh, ne, nh0, nhp, nhe0, nhep, nhepp)

      n_hi = nh0 * nh

    end subroutine


    subroutine iterate_ne(z, t, nh, ne, nh0, nhp, nhe0, nhep, nhepp)

      use atomic_rates, ONLY: YHELIUM, this_z

      double precision, intent (in   ) :: z, t, nh
      double precision, intent (inout) :: ne
      double precision, intent (  out) :: nh0, nhp, nhe0, nhep, nhepp

      integer :: i

      double precision, parameter :: xacc = 1.0d-8

      double precision :: f, df, eps
      double precision :: nhp_plus, nhep_plus, nhepp_plus
      double precision :: dnhp_dne, dnhep_dne, dnhepp_dne, dne

      ! Check if we have interpolated to this z
      if ( abs(z - this_z) .gt. xacc * z ) then
        STOP 'iterate_ne(): Wrong redshift!'
      endif

      i = 0
      ne = 1.0d0
      do  ! Newton-Raphson solver
        i = i + 1

        ! Ion number densities
        call ion_n(t, nh, ne, nhp, nhep, nhepp)

        ! Forward difference derivatives
        if (ne .gt. 0.0d0) then
          eps = xacc*ne
        else
          eps = 1.0d-24
        endif

        call ion_n(t, nh, (ne+eps), nhp_plus, nhep_plus, nhepp_plus)

        dnhp_dne   = (nhp_plus   - nhp)   / eps
        dnhep_dne  = (nhep_plus  - nhep)  / eps
        dnhepp_dne = (nhepp_plus - nhepp) / eps

        f   = ne - nhp - nhep - 2.0d0*nhepp
        df  = 1.0d0 - dnhp_dne - dnhep_dne - 2.0d0*dnhepp_dne
        dne = f/df

        ne = max((ne-dne), 0.0d0)

        if (abs(dne) < xacc) exit

        if (i .gt. 8) &
          print*, "ITERATION: ", i, " NUMBERS: ", z, t, ne, nhp, nhep, nhepp, df
        if (i .gt. 10) &
          STOP 'iterate_ne(): No convergence in Newton-Raphson!'
      enddo

      ! Get rates for the final ne
      call ion_n(t, nh, ne, nhp, nhep, nhepp)

      ! Neutral fractions:
      nh0   = 1.0d0 - nhp
      nhe0  = YHELIUM - (nhep + nhepp)

    end subroutine iterate_ne


    subroutine ion_n(t, nh, ne, nhp, nhep, nhepp)

      use atomic_rates, ONLY: YHELIUM, TCOOLMIN, TCOOLMAX, NCOOLTAB, &
                              AlphaHp, AlphaHep, AlphaHepp, Alphad, &
                              GammaeH0, GammaeHe0, GammaeHep, &
                              ggh0, gghe0, gghep

      double precision, intent(in   ) :: t, nh, ne
      double precision, intent(  out) :: nhp, nhep, nhepp
      double precision :: ahp, ahep, ahepp, ad, geh0, gehe0, gehep
      double precision :: ggh0ne, gghe0ne, gghepne
      double precision :: tmp, logT, deltaT, flo, fhi
      integer :: j

      logT = dlog10(t)
      if (logT .ge. TCOOLMAX) then ! Fully ionized plasma
        nhp   = 1.0d0
        nhep  = 0.0d0
        nhepp = YHELIUM
        return
      endif

      ! Temperature floor
      deltaT = (TCOOLMAX - TCOOLMIN) / NCOOLTAB
      if (logT .le. TCOOLMIN) logT = TCOOLMIN + 0.5d0 * deltaT

      ! Interpolate rates
      tmp = (logT - TCOOLMIN) / deltaT
      j = int(tmp)
      fhi = tmp - j
      flo = 1.0d0 - fhi
      j = j + 1  ! F90 arrays start with 1

      ahp   = flo*AlphaHp  (j) + fhi*AlphaHp  (j+1)
      ahep  = flo*AlphaHep (j) + fhi*AlphaHep (j+1)
      ahepp = flo*AlphaHepp(j) + fhi*AlphaHepp(j+1)
      ad    = flo*Alphad   (j) + fhi*Alphad   (j+1)
      geh0  = flo*GammaeH0 (j) + fhi*GammaeH0 (j+1)
      gehe0 = flo*GammaeHe0(j) + fhi*GammaeHe0(j+1)
      gehep = flo*GammaeHep(j) + fhi*GammaeHep(j+1)

      if (ne .gt. 0.0d0) then
        ggh0ne  = ggh0/ne/nh
        gghe0ne = gghe0/ne/nh
        gghepne = gghep/ne/nh
      else
        ggh0ne  = 0.0d0
        gghe0ne = 0.0d0
        gghepne = 0.0d0
      endif

      ! H+
      nhp = 1.0d0 - ahp/(ahp + geh0 + ggh0ne)

      ! He+
      if ((gehe0 + gghe0ne) .gt. 0.0d0) then
        nhep  = YHELIUM / ( 1.0d0 + (ahep + ad) / (gehe0 + gghe0ne) &
                            + (gehep + gghepne) / ahepp )
      else
        nhep  = 0.0d0
      endif

      ! He++
      if (nhep .gt. 0.0d0) then
        nhepp = nhep * (gehep + gghepne) / ahepp
      else
        nhepp = 0.0d0
      endif

    end subroutine ion_n

end module eos
