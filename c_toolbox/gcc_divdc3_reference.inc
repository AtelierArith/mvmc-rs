/* GNU runtime reference only; not compiled/invoked by Cargo.
 * Source https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/libgcc/libgcc2.c
 * Whole-source sha256=97d364b0d74d6bf073eb3829ab12884ce243c68af8f7601d84baade12ee15599
 * Extraction: double-mode macro branch + complete complex divide definition. */
/* More subroutines needed by GCC output code on some machines.  */
/* Compile this one with gcc.  */
/* Copyright (C) 1989-2023 Free Software Foundation, Inc.

This file is part of GCC.

GCC is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3, or (at your option) any later
version.

GCC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

Under Section 7 of GPL version 3, you are granted additional
permissions described in the GCC Runtime Library Exception, version
3.1, as published by the Free Software Foundation.

You should have received a copy of the GNU General Public License and
a copy of the GCC Runtime Library Exception along with this program;
see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
<http://www.gnu.org/licenses/>.  */

#elif defined(L_muldc3) || defined(L_divdc3)
# define MTYPE	DFtype
# define CTYPE	DCtype
# define MODE	dc
# define CEXT	__LIBGCC_DF_FUNC_EXT__
# define NOTRUNC (!__LIBGCC_DF_EXCESS_PRECISION__)
# define RBIG	(__LIBGCC_DF_MAX__ / 2)
# define RMIN	(__LIBGCC_DF_MIN__)
# define RMIN2	(__LIBGCC_DF_EPSILON__)
# define RMINSCAL (1 / __LIBGCC_DF_EPSILON__)
# define RMAX2  (RBIG * RMIN2)

CTYPE
CONCAT3(__div,MODE,3) (MTYPE a, MTYPE b, MTYPE c, MTYPE d)
{
#if defined(L_divhc3)						\
  || (defined(L_divsc3) && defined(__LIBGCC_HAVE_HWDBL__) )

  /* Half precision is handled with float precision.
     float is handled with double precision when double precision
     hardware is available.
     Due to the additional precision, the simple complex divide
     method (without Smith's method) is sufficient to get accurate
     answers and runs slightly faster than Smith's method.  */

  AMTYPE aa, bb, cc, dd;
  AMTYPE denom;
  MTYPE x, y;
  CTYPE res;
  aa = a;
  bb = b;
  cc = c;
  dd = d;

  denom = (cc * cc) + (dd * dd);
  x = ((aa * cc) + (bb * dd)) / denom;
  y = ((bb * cc) - (aa * dd)) / denom;

#else
  MTYPE denom, ratio, x, y;
  CTYPE res;

  /* double, extended, long double have significant potential
     underflow/overflow errors that can be greatly reduced with
     a limited number of tests and adjustments.  float is handled
     the same way when no HW double is available.
  */

  /* Scale by max(c,d) to reduce chances of denominator overflowing.  */
  if (FABS (c) < FABS (d))
    {
      /* Prevent underflow when denominator is near max representable.  */
      if (FABS (d) >= RBIG)
	{
	  a = a / 2;
	  b = b / 2;
	  c = c / 2;
	  d = d / 2;
	}
      /* Avoid overflow/underflow issues when c and d are small.
	 Scaling up helps avoid some underflows.
	 No new overflow possible since c&d < RMIN2.  */
      if (FABS (d) < RMIN2)
	{
	  a = a * RMINSCAL;
	  b = b * RMINSCAL;
	  c = c * RMINSCAL;
	  d = d * RMINSCAL;
	}
      else
	{
	  if (((FABS (a) < RMIN) && (FABS (b) < RMAX2) && (FABS (d) < RMAX2))
	      || ((FABS (b) < RMIN) && (FABS (a) < RMAX2)
		  && (FABS (d) < RMAX2)))
	    {
	      a = a * RMINSCAL;
	      b = b * RMINSCAL;
	      c = c * RMINSCAL;
	      d = d * RMINSCAL;
	    }
	}
      ratio = c / d;
      denom = (c * ratio) + d;
      /* Choose alternate order of computation if ratio is subnormal.  */
      if (FABS (ratio) > RMIN)
	{
	  x = ((a * ratio) + b) / denom;
	  y = ((b * ratio) - a) / denom;
	}
      else
	{
	  x = ((c * (a / d)) + b) / denom;
	  y = ((c * (b / d)) - a) / denom;
	}
    }
  else
    {
      /* Prevent underflow when denominator is near max representable.  */
      if (FABS (c) >= RBIG)
	{
	  a = a / 2;
	  b = b / 2;
	  c = c / 2;
	  d = d / 2;
	}
      /* Avoid overflow/underflow issues when both c and d are small.
	 Scaling up helps avoid some underflows.
	 No new overflow possible since both c&d are less than RMIN2.  */
      if (FABS (c) < RMIN2)
	{
	  a = a * RMINSCAL;
	  b = b * RMINSCAL;
	  c = c * RMINSCAL;
	  d = d * RMINSCAL;
	}
      else
	{
	  if (((FABS (a) < RMIN) && (FABS (b) < RMAX2) && (FABS (c) < RMAX2))
	      || ((FABS (b) < RMIN) && (FABS (a) < RMAX2)
		  && (FABS (c) < RMAX2)))
	    {
	      a = a * RMINSCAL;
	      b = b * RMINSCAL;
	      c = c * RMINSCAL;
	      d = d * RMINSCAL;
	    }
	}
      ratio = d / c;
      denom = (d * ratio) + c;
      /* Choose alternate order of computation if ratio is subnormal.  */
      if (FABS (ratio) > RMIN)
	{
	  x = ((b * ratio) + a) / denom;
	  y = (b - (a * ratio)) / denom;
	}
      else
	{
	  x = (a + (d * (b / c))) / denom;
	  y = (b - (d * (a / c))) / denom;
	}
    }
#endif

  /* Recover infinities and zeros that computed as NaN+iNaN; the only
     cases are nonzero/zero, infinite/finite, and finite/infinite.  */
  if (isnan (x) && isnan (y))
    {
      if (c == 0.0 && d == 0.0 && (!isnan (a) || !isnan (b)))
	{
	  x = COPYSIGN (INFINITY, c) * a;
	  y = COPYSIGN (INFINITY, c) * b;
	}
      else if ((isinf (a) || isinf (b)) && isfinite (c) && isfinite (d))
	{
	  a = COPYSIGN (isinf (a) ? 1 : 0, a);
	  b = COPYSIGN (isinf (b) ? 1 : 0, b);
	  x = INFINITY * (a * c + b * d);
	  y = INFINITY * (b * c - a * d);
	}
      else if ((isinf (c) || isinf (d)) && isfinite (a) && isfinite (b))
	{
	  c = COPYSIGN (isinf (c) ? 1 : 0, c);
	  d = COPYSIGN (isinf (d) ? 1 : 0, d);
	  x = 0.0 * (a * c + b * d);
	  y = 0.0 * (b * c - a * d);
	}
    }

  __real__ res = x;
  __imag__ res = y;
  return res;
}
#endif /* complex divide */
