/*
   File:          nnue.c
   Description:   Fast-Readout Pattern-Free Othello NNUE with ARM64 NEON Incremental Accumulator,
                  Accumulator-Embedded Corner/Quadrant Parity, and Bitboard Tactical Features (NNUE-002).
*/

#include "nnue.h"
#include "nnue_data.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NNUE_NEON 1
#else
#define USE_NNUE_NEON 0
#endif

static inline int
nnue_stage_bucket( int empties ) {
  if ( empties >= 17 )
    return 3;
  if ( empties >= 14 )
    return 2;
  if ( empties >= 11 )
    return 1;
  return 0;
}

static inline void
nnue_fill_tactical(
  const NnueTactical *tact,
  int16_t f10[8],
  int16_t *out_tf0_tf4
) {
  if ( tact ) {
    int om = tact->opp_mob;
    int oc = tact->opp_corner_moves;
    int fl = tact->flipped;
    int om_c = ( om < 0 ) ? 0 : ( ( om > 16 ) ? 16 : om );
    int oc_c = ( oc < 0 ) ? 0 : ( ( oc > 4 ) ? 4 : oc );
    int fl_c = ( fl < 0 ) ? 0 : ( ( fl > 16 ) ? 16 : fl );
    int16_t tf0 = (int16_t)( om_c << 3 );
    int16_t tf4 = (int16_t)( ( tact->quad_parity ? 1 : 0 ) << 7 );
    f10[0] = tf0;
    f10[1] = (int16_t)( oc_c << 6 );
    f10[2] = (int16_t)( ( tact->is_corner ? 1 : 0 ) << 7 );
    f10[3] = (int16_t)( ( tact->is_open_x_or_c ? 1 : 0 ) << 7 );
    f10[4] = tf4;
    f10[5] = (int16_t)( fl_c << 3 );
    f10[6] = (int16_t)( ( om == 0 ? 1 : 0 ) << 7 );
    f10[7] = (int16_t)( ( (int32_t)tf0 * (int32_t)tf0 ) >> 7 );
    *out_tf0_tf4 = (int16_t)( ( (int32_t)tf0 * (int32_t)tf4 ) >> 7 );
  } else {
    for ( int k = 0; k < 8; k++ )
      f10[k] = 0;
    *out_tf0_tf4 = 0;
  }
}

void
nnue_init_acc_scalar( NnueAcc *acc, uint64_t n_my, uint64_t n_opp ) {
  for ( int k = 0; k < NNUE_ACC_DIM; k++ )
    acc->v[k] = nnue_bias[k];

  uint64_t m = n_my;
  while ( m ) {
    int sq = __builtin_ctzll( m );
    m &= m - 1;
    const int16_t *w = nnue_w_my[sq];
    for ( int k = 0; k < NNUE_ACC_DIM; k++ )
      acc->v[k] = (int16_t)( acc->v[k] + w[k] );
  }

  uint64_t o = n_opp;
  while ( o ) {
    int sq = __builtin_ctzll( o );
    o &= o - 1;
    const int16_t *w = nnue_w_opp[sq];
    for ( int k = 0; k < NNUE_ACC_DIM; k++ )
      acc->v[k] = (int16_t)( acc->v[k] + w[k] );
  }
}

void
nnue_init_acc( NnueAcc *acc, uint64_t n_my, uint64_t n_opp ) {
#if USE_NNUE_NEON && ( NNUE_ACC_DIM == 64 )
  int16x8_t a0 = vld1q_s16( &nnue_bias[0] );
  int16x8_t a1 = vld1q_s16( &nnue_bias[8] );
  int16x8_t a2 = vld1q_s16( &nnue_bias[16] );
  int16x8_t a3 = vld1q_s16( &nnue_bias[24] );
  int16x8_t a4 = vld1q_s16( &nnue_bias[32] );
  int16x8_t a5 = vld1q_s16( &nnue_bias[40] );
  int16x8_t a6 = vld1q_s16( &nnue_bias[48] );
  int16x8_t a7 = vld1q_s16( &nnue_bias[56] );

  uint64_t m = n_my;
  while ( m ) {
    int sq = __builtin_ctzll( m );
    m &= m - 1;
    const int16_t *w = nnue_w_my[sq];
    a0 = vaddq_s16( a0, vld1q_s16( &w[0] ) );
    a1 = vaddq_s16( a1, vld1q_s16( &w[8] ) );
    a2 = vaddq_s16( a2, vld1q_s16( &w[16] ) );
    a3 = vaddq_s16( a3, vld1q_s16( &w[24] ) );
    a4 = vaddq_s16( a4, vld1q_s16( &w[32] ) );
    a5 = vaddq_s16( a5, vld1q_s16( &w[40] ) );
    a6 = vaddq_s16( a6, vld1q_s16( &w[48] ) );
    a7 = vaddq_s16( a7, vld1q_s16( &w[56] ) );
  }

  uint64_t o = n_opp;
  while ( o ) {
    int sq = __builtin_ctzll( o );
    o &= o - 1;
    const int16_t *w = nnue_w_opp[sq];
    a0 = vaddq_s16( a0, vld1q_s16( &w[0] ) );
    a1 = vaddq_s16( a1, vld1q_s16( &w[8] ) );
    a2 = vaddq_s16( a2, vld1q_s16( &w[16] ) );
    a3 = vaddq_s16( a3, vld1q_s16( &w[24] ) );
    a4 = vaddq_s16( a4, vld1q_s16( &w[32] ) );
    a5 = vaddq_s16( a5, vld1q_s16( &w[40] ) );
    a6 = vaddq_s16( a6, vld1q_s16( &w[48] ) );
    a7 = vaddq_s16( a7, vld1q_s16( &w[56] ) );
  }

  vst1q_s16( &acc->v[0], a0 );
  vst1q_s16( &acc->v[8], a1 );
  vst1q_s16( &acc->v[16], a2 );
  vst1q_s16( &acc->v[24], a3 );
  vst1q_s16( &acc->v[32], a4 );
  vst1q_s16( &acc->v[40], a5 );
  vst1q_s16( &acc->v[48], a6 );
  vst1q_s16( &acc->v[56], a7 );
#else
  nnue_init_acc_scalar( acc, n_my, n_opp );
#endif
}

void
nnue_init_parent_base_acc( NnueAcc *base_acc, uint64_t p_my, uint64_t p_opp ) {
  nnue_init_acc( base_acc, p_my, p_opp );
}

void
nnue_update_child_acc(
  NnueAcc *child_acc,
  const NnueAcc *base_acc,
  int sq,
  uint64_t flips
) {
#if USE_NNUE_NEON && ( NNUE_ACC_DIM == 64 )
  const int16_t *w_sq = nnue_w_my[sq];
  int16x8_t a0 = vaddq_s16( vld1q_s16( &base_acc->v[0] ), vld1q_s16( &w_sq[0] ) );
  int16x8_t a1 = vaddq_s16( vld1q_s16( &base_acc->v[8] ), vld1q_s16( &w_sq[8] ) );
  int16x8_t a2 = vaddq_s16( vld1q_s16( &base_acc->v[16] ), vld1q_s16( &w_sq[16] ) );
  int16x8_t a3 = vaddq_s16( vld1q_s16( &base_acc->v[24] ), vld1q_s16( &w_sq[24] ) );
  int16x8_t a4 = vaddq_s16( vld1q_s16( &base_acc->v[32] ), vld1q_s16( &w_sq[32] ) );
  int16x8_t a5 = vaddq_s16( vld1q_s16( &base_acc->v[40] ), vld1q_s16( &w_sq[40] ) );
  int16x8_t a6 = vaddq_s16( vld1q_s16( &base_acc->v[48] ), vld1q_s16( &w_sq[48] ) );
  int16x8_t a7 = vaddq_s16( vld1q_s16( &base_acc->v[56] ), vld1q_s16( &w_sq[56] ) );

  uint64_t f = flips;
  while ( f ) {
    int fsq = __builtin_ctzll( f );
    f &= f - 1;
    const int16_t *wf = nnue_w_flip[fsq];
    a0 = vaddq_s16( a0, vld1q_s16( &wf[0] ) );
    a1 = vaddq_s16( a1, vld1q_s16( &wf[8] ) );
    a2 = vaddq_s16( a2, vld1q_s16( &wf[16] ) );
    a3 = vaddq_s16( a3, vld1q_s16( &wf[24] ) );
    a4 = vaddq_s16( a4, vld1q_s16( &wf[32] ) );
    a5 = vaddq_s16( a5, vld1q_s16( &wf[40] ) );
    a6 = vaddq_s16( a6, vld1q_s16( &wf[48] ) );
    /* Note: lanes 56..63 (a7) have nnue_w_flip[fsq][56..63] == 0 for all 64 squares! */
  }

  vst1q_s16( &child_acc->v[0], a0 );
  vst1q_s16( &child_acc->v[8], a1 );
  vst1q_s16( &child_acc->v[16], a2 );
  vst1q_s16( &child_acc->v[24], a3 );
  vst1q_s16( &child_acc->v[32], a4 );
  vst1q_s16( &child_acc->v[40], a5 );
  vst1q_s16( &child_acc->v[48], a6 );
  vst1q_s16( &child_acc->v[56], a7 );
#else
  const int16_t *w_sq = nnue_w_my[sq];
  for ( int k = 0; k < NNUE_ACC_DIM; k++ )
    child_acc->v[k] = (int16_t)( base_acc->v[k] + w_sq[k] );

  uint64_t f = flips;
  while ( f ) {
    int fsq = __builtin_ctzll( f );
    f &= f - 1;
    const int16_t *wf = nnue_w_flip[fsq];
    for ( int k = 0; k < 56; k++ )
      child_acc->v[k] = (int16_t)( child_acc->v[k] + wf[k] );
  }
#endif
}

void
nnue_evaluate_acc_scalar(
  const NnueAcc *acc,
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties,
  int32_t *out_raw_order,
  int32_t *out_raw_value
) {
  (void)n_my;
  (void)n_opp;
  int16_t vp[7][8];
  int16_t vn[7][8];
  int16_t r[7][8];
  int16_t feats[NNUE_FEAT_DIM];

  for ( int c = 0; c < 7; c++ ) {
    int32_t sum_p = 0;
    int32_t sum_n = 0;
    int32_t sum_sp = 0;
    int32_t sum_sn = 0;
    for ( int g = 0; g < 8; g++ ) {
      int16_t x = acc->v[c * 8 + g];
      int16_t p = ( x > 0 ) ? ( ( x > 128 ) ? 128 : x ) : 0;
      int16_t n = ( -x > 0 ) ? ( ( -x > 128 ) ? 128 : -x ) : 0;
      vp[c][g] = p;
      vn[c][g] = n;
      r[c][g] = (int16_t)( p - n );
      sum_p += p;
      sum_n += n;
      sum_sp += ( (int32_t)p * (int32_t)p ) >> 7;
      sum_sn += ( (int32_t)n * (int32_t)n ) >> 7;
    }
    feats[0 + c] = (int16_t)sum_p;
    feats[8 + c] = (int16_t)sum_n;
    feats[16 + c] = (int16_t)sum_sp;
    feats[24 + c] = (int16_t)sum_sn;
  }

  for ( int c = 0; c < 7; c++ ) {
    int c1 = ( c + 1 < 7 ) ? ( c + 1 ) : 0;
    int32_t sum_cr = 0;
    for ( int g = 0; g < 8; g++ )
      sum_cr += ( (int32_t)r[c][g] * (int32_t)r[c1][g] ) >> 7;
    feats[32 + c] = (int16_t)sum_cr;
  }

  /* Extract parity vectors from acc->v[56..63] (`a7`) */
  int sum_open_odd = 0;
  int sum_open_even = 0;
  int sum_c33_odd = 0;
  int sum_quad_odd = 0;
  int sum_c33_eq1 = 0;
  int sum_quad_eq1 = 0;
  int sum_c33_eq2 = 0;
  int sum_quad_eq2 = 0;
  int sum_cemp_lo = 0;

  for ( int c = 0; c < 4; c++ ) {
    int16_t v_lo = acc->v[56 + c];
    int16_t v_hi = acc->v[60 + c];
    int cnt_lo = v_lo & 31;
    int odd_lo = v_lo & 1;
    int cemp_lo = v_lo >> 5;
    int cnt_hi = v_hi & 31;
    int odd_hi = v_hi & 1;

    sum_open_odd += ( cemp_lo & odd_lo );
    sum_open_even += ( cemp_lo & ( 1 - odd_lo ) );
    sum_c33_odd += odd_lo;
    sum_quad_odd += odd_hi;
    sum_c33_eq1 += ( cnt_lo == 1 );
    sum_quad_eq1 += ( cnt_hi == 1 );
    sum_c33_eq2 += ( cnt_lo == 2 );
    sum_quad_eq2 += ( cnt_hi == 2 );
    sum_cemp_lo += cemp_lo;
  }

  feats[7] = (int16_t)( sum_open_odd << 6 );
  feats[15] = (int16_t)( sum_open_even << 6 );
  feats[23] = (int16_t)( ( sum_c33_odd + sum_quad_odd ) << 5 );
  feats[31] = (int16_t)( ( sum_c33_eq1 + sum_quad_eq1 ) << 5 );
  feats[39] = (int16_t)( ( sum_c33_eq2 + sum_quad_eq2 ) << 5 );

  int16_t sr_s[8];
  int16_t sa_s[8];
  for ( int k = 0; k < 8; k++ ) {
    sr_s[k] = (int16_t)( ( feats[0 + k] - feats[8 + k] ) >> 3 );
    sa_s[k] = (int16_t)( ( feats[0 + k] + feats[8 + k] ) >> 3 );
  }

  int16_t tf0_tf4 = 0;
  nnue_fill_tactical( tact, &feats[80], &tf0_tf4 );
  int16_t tf0 = feats[80];

  for ( int k = 0; k < 8; k++ ) {
    int k1 = ( k + 1 ) & 7;
    feats[40 + k] = (int16_t)( ( (int32_t)sr_s[k] * (int32_t)sr_s[k] ) >> 4 );
    feats[48 + k] = (int16_t)( ( (int32_t)sa_s[k] * (int32_t)sa_s[k] ) >> 4 );
    feats[56 + k] = (int16_t)( ( (int32_t)sr_s[k] * (int32_t)sr_s[k1] ) >> 4 );
    feats[64 + k] = (int16_t)( ( (int32_t)sr_s[k] * (int32_t)tf0 ) >> 6 );
  }

  feats[72] = (int16_t)( sum_c33_odd << 5 );
  feats[73] = (int16_t)( sum_quad_odd << 5 );
  feats[74] = (int16_t)( sum_c33_eq1 << 5 );
  feats[75] = (int16_t)( sum_quad_eq1 << 5 );
  feats[76] = (int16_t)( sum_c33_eq2 << 5 );
  feats[77] = (int16_t)( sum_quad_eq2 << 5 );
  feats[78] = (int16_t)( sum_cemp_lo << 5 );
  feats[79] = tf0_tf4;

  int b = nnue_stage_bucket( empties );
  const int16_t *ow = nnue_order_weight[b];
  const int16_t *vw = nnue_value_weight[b];
  int32_t ord_sum = nnue_order_bias[b];
  int32_t val_sum = nnue_value_bias[b];

  for ( int k = 0; k < NNUE_FEAT_DIM; k++ ) {
    ord_sum += (int32_t)feats[k] * (int32_t)ow[k];
    val_sum += (int32_t)feats[k] * (int32_t)vw[k];
  }

  if ( out_raw_order )
    *out_raw_order = ord_sum;
  if ( out_raw_value )
    *out_raw_value = val_sum;
}

#if USE_NNUE_NEON && ( NNUE_FEAT_DIM == 88 )
static inline int16x8_t
hsum8_s16(
  int16x8_t v0, int16x8_t v1, int16x8_t v2, int16x8_t v3,
  int16x8_t v4, int16x8_t v5, int16x8_t v6, int16x8_t v7
) {
  int16x8_t p01 = vpaddq_s16( v0, v1 );
  int16x8_t p23 = vpaddq_s16( v2, v3 );
  int16x8_t p45 = vpaddq_s16( v4, v5 );
  int16x8_t p67 = vpaddq_s16( v6, v7 );
  int16x8_t p0123 = vpaddq_s16( p01, p23 );
  int16x8_t p4567 = vpaddq_s16( p45, p67 );
  return vpaddq_s16( p0123, p4567 );
}

static inline int16x8_t
hsum4_halves_s16(
  int16x8_t v0, int16x8_t v1, int16x8_t v2, int16x8_t v3
) {
  int16x8_t p01 = vpaddq_s16( v0, v1 );
  int16x8_t p23 = vpaddq_s16( v2, v3 );
  return vpaddq_s16( p01, p23 );
}

#define NNUE_BUILD_FEATURE_VECTORS() \
  int16_t t_f10[8] __attribute__((aligned(16))); \
  int16_t tf0_tf4 = 0; \
  nnue_fill_tactical( tact, t_f10, &tf0_tf4 ); \
  int16x8_t f10 = vld1q_s16( t_f10 ); \
  const int16x8_t z = vdupq_n_s16( 0 ); \
  const int16x8_t q128 = vdupq_n_s16( 128 ); \
  int16x8_t vp0 = vminq_s16( vmaxq_s16( a0, z ), q128 ), vn0 = vminq_s16( vmaxq_s16( vnegq_s16( a0 ), z ), q128 ); \
  int16x8_t vp1 = vminq_s16( vmaxq_s16( a1, z ), q128 ), vn1 = vminq_s16( vmaxq_s16( vnegq_s16( a1 ), z ), q128 ); \
  int16x8_t vp2 = vminq_s16( vmaxq_s16( a2, z ), q128 ), vn2 = vminq_s16( vmaxq_s16( vnegq_s16( a2 ), z ), q128 ); \
  int16x8_t vp3 = vminq_s16( vmaxq_s16( a3, z ), q128 ), vn3 = vminq_s16( vmaxq_s16( vnegq_s16( a3 ), z ), q128 ); \
  int16x8_t vp4 = vminq_s16( vmaxq_s16( a4, z ), q128 ), vn4 = vminq_s16( vmaxq_s16( vnegq_s16( a4 ), z ), q128 ); \
  int16x8_t vp5 = vminq_s16( vmaxq_s16( a5, z ), q128 ), vn5 = vminq_s16( vmaxq_s16( vnegq_s16( a5 ), z ), q128 ); \
  int16x8_t vp6 = vminq_s16( vmaxq_s16( a6, z ), q128 ), vn6 = vminq_s16( vmaxq_s16( vnegq_s16( a6 ), z ), q128 ); \
  int16x8_t sp0 = vshrq_n_s16( vmulq_s16( vp0, vp0 ), 7 ), sn0 = vshrq_n_s16( vmulq_s16( vn0, vn0 ), 7 ); \
  int16x8_t sp1 = vshrq_n_s16( vmulq_s16( vp1, vp1 ), 7 ), sn1 = vshrq_n_s16( vmulq_s16( vn1, vn1 ), 7 ); \
  int16x8_t sp2 = vshrq_n_s16( vmulq_s16( vp2, vp2 ), 7 ), sn2 = vshrq_n_s16( vmulq_s16( vn2, vn2 ), 7 ); \
  int16x8_t sp3 = vshrq_n_s16( vmulq_s16( vp3, vp3 ), 7 ), sn3 = vshrq_n_s16( vmulq_s16( vn3, vn3 ), 7 ); \
  int16x8_t sp4 = vshrq_n_s16( vmulq_s16( vp4, vp4 ), 7 ), sn4 = vshrq_n_s16( vmulq_s16( vn4, vn4 ), 7 ); \
  int16x8_t sp5 = vshrq_n_s16( vmulq_s16( vp5, vp5 ), 7 ), sn5 = vshrq_n_s16( vmulq_s16( vn5, vn5 ), 7 ); \
  int16x8_t sp6 = vshrq_n_s16( vmulq_s16( vp6, vp6 ), 7 ), sn6 = vshrq_n_s16( vmulq_s16( vn6, vn6 ), 7 ); \
  int16x8_t r0 = vsubq_s16( vp0, vn0 ), r1 = vsubq_s16( vp1, vn1 ), r2 = vsubq_s16( vp2, vn2 ), r3 = vsubq_s16( vp3, vn3 ); \
  int16x8_t r4 = vsubq_s16( vp4, vn4 ), r5 = vsubq_s16( vp5, vn5 ), r6 = vsubq_s16( vp6, vn6 ); \
  int16x8_t cr0 = vshrq_n_s16( vmulq_s16( r0, r1 ), 7 ), cr1 = vshrq_n_s16( vmulq_s16( r1, r2 ), 7 ); \
  int16x8_t cr2 = vshrq_n_s16( vmulq_s16( r2, r3 ), 7 ), cr3 = vshrq_n_s16( vmulq_s16( r3, r4 ), 7 ); \
  int16x8_t cr4 = vshrq_n_s16( vmulq_s16( r4, r5 ), 7 ), cr5 = vshrq_n_s16( vmulq_s16( r5, r6 ), 7 ); \
  int16x8_t cr6 = vshrq_n_s16( vmulq_s16( r6, r0 ), 7 ); \
  const int16x8_t one = vdupq_n_s16( 1 ); \
  const int16x8_t two = vdupq_n_s16( 2 ); \
  const int16x8_t mask31 = vdupq_n_s16( 31 ); \
  int16x8_t cnt_vec = vandq_s16( a7, mask31 ); \
  int16x8_t odd_vec = vandq_s16( a7, one ); \
  int16x8_t cemp_vec = vshrq_n_s16( a7, 5 ); \
  int16x8_t open_odd_vec = vandq_s16( cemp_vec, odd_vec ); \
  int16x8_t open_even_vec = vbicq_s16( cemp_vec, odd_vec ); \
  int16x8_t eq1_vec = vandq_s16( vreinterpretq_s16_u16( vceqq_s16( cnt_vec, one ) ), one ); \
  int16x8_t eq2_vec = vandq_s16( vreinterpretq_s16_u16( vceqq_s16( cnt_vec, two ) ), one ); \
  int16x8_t f0 = hsum8_s16( vp0, vp1, vp2, vp3, vp4, vp5, vp6, vshlq_n_s16( open_odd_vec, 6 ) ); \
  int16x8_t f1 = hsum8_s16( vn0, vn1, vn2, vn3, vn4, vn5, vn6, vshlq_n_s16( open_even_vec, 6 ) ); \
  int16x8_t f2 = hsum8_s16( sp0, sp1, sp2, sp3, sp4, sp5, sp6, vshlq_n_s16( odd_vec, 5 ) ); \
  int16x8_t f3 = hsum8_s16( sn0, sn1, sn2, sn3, sn4, sn5, sn6, vshlq_n_s16( eq1_vec, 5 ) ); \
  int16x8_t f4 = hsum8_s16( cr0, cr1, cr2, cr3, cr4, cr5, cr6, vshlq_n_s16( eq2_vec, 5 ) ); \
  int16x8_t sr_s = vshrq_n_s16( vsubq_s16( f0, f1 ), 3 ); \
  int16x8_t sa_s = vshrq_n_s16( vaddq_s16( f0, f1 ), 3 ); \
  int16x8_t sr_rot1 = vextq_s16( sr_s, sr_s, 1 ); \
  int16x8_t f5 = vshrq_n_s16( vmulq_s16( sr_s, sr_s ), 4 ); \
  int16x8_t f6 = vshrq_n_s16( vmulq_s16( sa_s, sa_s ), 4 ); \
  int16x8_t f7 = vshrq_n_s16( vmulq_s16( sr_s, sr_rot1 ), 4 ); \
  int16x8_t f8 = vshrq_n_s16( vmulq_laneq_s16( sr_s, f10, 0 ), 6 ); \
  int16x8_t f9_raw = vshlq_n_s16( hsum4_halves_s16( odd_vec, eq1_vec, eq2_vec, cemp_vec ), 5 ); \
  int16x8_t f9 = vsetq_lane_s16( tf0_tf4, f9_raw, 7 );

#define NNUE_DOT11(w_ptr, sum_out) do { \
  int32x4_t s0 = vdupq_n_s32( 0 ); \
  int32x4_t s1 = vdupq_n_s32( 0 ); \
  int16x8_t wv; \
  wv = vld1q_s16( &(w_ptr)[0] );  s0 = vmlal_s16( s0, vget_low_s16( f0 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f0 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[8] );  s0 = vmlal_s16( s0, vget_low_s16( f1 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f1 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[16] ); s0 = vmlal_s16( s0, vget_low_s16( f2 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f2 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[24] ); s0 = vmlal_s16( s0, vget_low_s16( f3 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f3 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[32] ); s0 = vmlal_s16( s0, vget_low_s16( f4 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f4 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[40] ); s0 = vmlal_s16( s0, vget_low_s16( f5 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f5 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[48] ); s0 = vmlal_s16( s0, vget_low_s16( f6 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f6 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[56] ); s0 = vmlal_s16( s0, vget_low_s16( f7 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f7 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[64] ); s0 = vmlal_s16( s0, vget_low_s16( f8 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f8 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[72] ); s0 = vmlal_s16( s0, vget_low_s16( f9 ),  vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f9 ),  vget_high_s16( wv ) ); \
  wv = vld1q_s16( &(w_ptr)[80] ); s0 = vmlal_s16( s0, vget_low_s16( f10 ), vget_low_s16( wv ) ); s1 = vmlal_s16( s1, vget_high_s16( f10 ), vget_high_s16( wv ) ); \
  sum_out = vaddvq_s32( vaddq_s32( s0, s1 ) ); \
} while ( 0 )
#endif

void
nnue_evaluate_acc(
  const NnueAcc *acc,
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties,
  int32_t *out_raw_order,
  int32_t *out_raw_value
) {
#if USE_NNUE_NEON && ( NNUE_FEAT_DIM == 88 )
  (void)n_my;
  (void)n_opp;
  int16x8_t a0 = vld1q_s16( &acc->v[0] );
  int16x8_t a1 = vld1q_s16( &acc->v[8] );
  int16x8_t a2 = vld1q_s16( &acc->v[16] );
  int16x8_t a3 = vld1q_s16( &acc->v[24] );
  int16x8_t a4 = vld1q_s16( &acc->v[32] );
  int16x8_t a5 = vld1q_s16( &acc->v[40] );
  int16x8_t a6 = vld1q_s16( &acc->v[48] );
  int16x8_t a7 = vld1q_s16( &acc->v[56] );

  NNUE_BUILD_FEATURE_VECTORS();

  int b = nnue_stage_bucket( empties );
  if ( out_raw_order ) {
    const int16_t *ow = nnue_order_weight[b];
    int32_t dot_o = 0;
    NNUE_DOT11( ow, dot_o );
    *out_raw_order = nnue_order_bias[b] + dot_o;
  }
  if ( out_raw_value ) {
    const int16_t *vw = nnue_value_weight[b];
    int32_t dot_v = 0;
    NNUE_DOT11( vw, dot_v );
    *out_raw_value = nnue_value_bias[b] + dot_v;
  }
#else
  nnue_evaluate_acc_scalar( acc, n_my, n_opp, tact, empties, out_raw_order, out_raw_value );
#endif
}

int
nnue_evaluate_order_acc(
  const NnueAcc *acc,
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties
) {
  int32_t raw_ord = 0;
  nnue_evaluate_acc( acc, n_my, n_opp, tact, empties, &raw_ord, 0 );
  return (int)( raw_ord >> 6 );
}

int
nnue_evaluate_value_acc(
  const NnueAcc *acc,
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties
) {
  int32_t raw_val = 0;
  nnue_evaluate_acc( acc, n_my, n_opp, tact, empties, 0, &raw_val );
  return (int)( raw_val >> 6 );
}

int
nnue_evaluate_value_board(
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties
) {
#if USE_NNUE_NEON && ( NNUE_FEAT_DIM == 88 )
  int16x8_t a0 = vld1q_s16( &nnue_bias[0] );
  int16x8_t a1 = vld1q_s16( &nnue_bias[8] );
  int16x8_t a2 = vld1q_s16( &nnue_bias[16] );
  int16x8_t a3 = vld1q_s16( &nnue_bias[24] );
  int16x8_t a4 = vld1q_s16( &nnue_bias[32] );
  int16x8_t a5 = vld1q_s16( &nnue_bias[40] );
  int16x8_t a6 = vld1q_s16( &nnue_bias[48] );
  int16x8_t a7 = vld1q_s16( &nnue_bias[56] );

  uint64_t m = n_my;
  while ( m ) {
    int sq = __builtin_ctzll( m );
    m &= m - 1;
    const int16_t *w = nnue_w_my[sq];
    a0 = vaddq_s16( a0, vld1q_s16( &w[0] ) );
    a1 = vaddq_s16( a1, vld1q_s16( &w[8] ) );
    a2 = vaddq_s16( a2, vld1q_s16( &w[16] ) );
    a3 = vaddq_s16( a3, vld1q_s16( &w[24] ) );
    a4 = vaddq_s16( a4, vld1q_s16( &w[32] ) );
    a5 = vaddq_s16( a5, vld1q_s16( &w[40] ) );
    a6 = vaddq_s16( a6, vld1q_s16( &w[48] ) );
    a7 = vaddq_s16( a7, vld1q_s16( &w[56] ) );
  }

  uint64_t o = n_opp;
  while ( o ) {
    int sq = __builtin_ctzll( o );
    o &= o - 1;
    const int16_t *w = nnue_w_opp[sq];
    a0 = vaddq_s16( a0, vld1q_s16( &w[0] ) );
    a1 = vaddq_s16( a1, vld1q_s16( &w[8] ) );
    a2 = vaddq_s16( a2, vld1q_s16( &w[16] ) );
    a3 = vaddq_s16( a3, vld1q_s16( &w[24] ) );
    a4 = vaddq_s16( a4, vld1q_s16( &w[32] ) );
    a5 = vaddq_s16( a5, vld1q_s16( &w[40] ) );
    a6 = vaddq_s16( a6, vld1q_s16( &w[48] ) );
    a7 = vaddq_s16( a7, vld1q_s16( &w[56] ) );
  }

  NNUE_BUILD_FEATURE_VECTORS();

  int b = nnue_stage_bucket( empties );
  const int16_t *vw = nnue_value_weight[b];
  int32_t dot_v = 0;
  NNUE_DOT11( vw, dot_v );
  return (int)( ( nnue_value_bias[b] + dot_v ) >> 6 );
#else
  NnueAcc acc;
  nnue_init_acc( &acc, n_my, n_opp );
  return nnue_evaluate_value_acc( &acc, n_my, n_opp, tact, empties );
#endif
}

int
nnue_verify_reference_cases( void ) {
  for ( int i = 0; i < NNUE_REF_CASE_COUNT; i++ ) {
    NnueAcc acc_s, acc_n;
    uint64_t n_my = nnue_ref_cases[i].n_my;
    uint64_t n_opp = nnue_ref_cases[i].n_opp;
    int empties = nnue_ref_cases[i].empties;
    NnueTactical tact = {
      nnue_ref_cases[i].opp_mob,
      nnue_ref_cases[i].opp_corner_moves,
      nnue_ref_cases[i].is_corner,
      nnue_ref_cases[i].is_open_x_or_c,
      nnue_ref_cases[i].quad_parity,
      nnue_ref_cases[i].flipped
    };

    nnue_init_acc_scalar( &acc_s, n_my, n_opp );
    nnue_init_acc( &acc_n, n_my, n_opp );
    for ( int k = 0; k < NNUE_ACC_DIM; k++ ) {
      if ( acc_s.v[k] != acc_n.v[k] )
        return 0;
    }
    int32_t ord_s = 0, val_s = 0;
    int32_t ord_n = 0, val_n = 0;
    nnue_evaluate_acc_scalar( &acc_s, n_my, n_opp, &tact, empties, &ord_s, &val_s );
    nnue_evaluate_acc( &acc_n, n_my, n_opp, &tact, empties, &ord_n, &val_n );
    if ( ord_s != nnue_ref_cases[i].expected_raw_order ||
         val_s != nnue_ref_cases[i].expected_raw_value ||
         ord_n != nnue_ref_cases[i].expected_raw_order ||
         val_n != nnue_ref_cases[i].expected_raw_value ) {
      return 0;
    }
  }
  return 1;
}
