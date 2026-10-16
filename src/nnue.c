/*
   File:          nnue.c
   Description:   E-Unified Pattern-Free Othello NNUE with ARM64 NEON Incremental Accumulator,
                  Corner/Quadrant Empty Parity, and Zero-Cost Bitboard Tactical Features (Othello NNUE Training & Export).
*/

#include "nnue.h"
#include "nnue_data.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NNUE_NEON 1
#else
#define USE_NNUE_NEON 0
#endif

static const uint64_t NNUE_C33_MASKS[4] = {
  0x0000000000070707ULL,
  0x0000000000E0E0E0ULL,
  0x0707070000000000ULL,
  0xE0E0E00000000000ULL
};

static const uint64_t NNUE_QUAD_MASKS[4] = {
  0x000000000F0F0F0FULL,
  0x00000000F0F0F0F0ULL,
  0x0F0F0F0F00000000ULL,
  0xF0F0F0F000000000ULL
};

static const int NNUE_CORNER_BITS[4] = { 0, 7, 56, 63 };
static const int NNUE_X_BITS[4] = { 9, 14, 49, 54 };

static inline int
nnue_stage_bucket( int empties ) {
  if ( empties >= 22 )
    return 4;
  if ( empties >= 18 )
    return 3;
  if ( empties >= 14 )
    return 2;
  if ( empties >= 11 )
    return 1;
  return 0;
}

static inline void
nnue_compute_parity_and_tactical(
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int16_t pf[16],
  int16_t tf[8],
  int16_t ts[8]
) {
  uint64_t emp = ~( n_my | n_opp );
  int sum_odd = 0;
  int sum_open_odd = 0;
  int sum_open_even = 0;
  int sum_cnt1 = 0;
  int sum_cnt2 = 0;
  int sum_cnt3 = 0;
  int sum_qodd = 0;
  int sum_qcnt1 = 0;
  int sum_c_emp = 0;
  int sum_open_my_x = 0;
  int sum_open_opp_x = 0;
  int sum_my_corn = 0;
  int sum_opp_corn = 0;
  int sum_open_odd_my_x = 0;
  int sum_open_odd_opp_x = 0;

  for ( int c = 0; c < 4; c++ ) {
    int cnt = __builtin_popcountll( emp & NNUE_C33_MASKS[c] );
    int odd = cnt & 1;
    int qcnt = __builtin_popcountll( emp & NNUE_QUAD_MASKS[c] );
    int qodd = qcnt & 1;
    int cbit = NNUE_CORNER_BITS[c];
    int xbit = NNUE_X_BITS[c];
    int c_emp = (int)( ( emp >> cbit ) & 1ULL );
    int my_x = (int)( ( n_my >> xbit ) & 1ULL );
    int opp_x = (int)( ( n_opp >> xbit ) & 1ULL );
    int my_corn = (int)( ( n_my >> cbit ) & 1ULL );
    int opp_corn = (int)( ( n_opp >> cbit ) & 1ULL );

    sum_odd += odd;
    sum_open_odd += ( c_emp & odd );
    sum_open_even += ( c_emp & ( 1 - odd ) );
    sum_cnt1 += ( cnt == 1 );
    sum_cnt2 += ( cnt == 2 );
    sum_cnt3 += ( cnt == 3 );
    sum_qodd += qodd;
    sum_qcnt1 += ( qcnt == 1 );
    sum_c_emp += c_emp;
    sum_open_my_x += ( c_emp & my_x );
    sum_open_opp_x += ( c_emp & opp_x );
    sum_my_corn += my_corn;
    sum_opp_corn += opp_corn;
    sum_open_odd_my_x += ( c_emp & odd & my_x );
    sum_open_odd_opp_x += ( c_emp & odd & opp_x );
  }

  int disc_diff = __builtin_popcountll( n_my ) - __builtin_popcountll( n_opp );

  pf[0] = (int16_t)( sum_odd << 5 );
  pf[1] = (int16_t)( sum_open_odd << 5 );
  pf[2] = (int16_t)( sum_open_even << 5 );
  pf[3] = (int16_t)( sum_cnt1 << 5 );
  pf[4] = (int16_t)( sum_cnt2 << 5 );
  pf[5] = (int16_t)( sum_cnt3 << 5 );
  pf[6] = (int16_t)( sum_qodd << 5 );
  pf[7] = (int16_t)( sum_qcnt1 << 5 );
  pf[8] = (int16_t)( sum_c_emp << 5 );
  pf[9] = (int16_t)( sum_open_my_x << 5 );
  pf[10] = (int16_t)( sum_open_opp_x << 5 );
  pf[11] = (int16_t)( sum_my_corn << 5 );
  pf[12] = (int16_t)( sum_opp_corn << 5 );
  pf[13] = (int16_t)( disc_diff << 2 );
  pf[14] = (int16_t)( sum_open_odd_my_x << 5 );
  pf[15] = (int16_t)( sum_open_odd_opp_x << 5 );

  if ( tact ) {
    int om = tact->opp_mob;
    int oc = tact->opp_corner_moves;
    int fl = tact->flipped;
    int om_c = ( om < 0 ) ? 0 : ( ( om > 16 ) ? 16 : om );
    int oc_c = ( oc < 0 ) ? 0 : ( ( oc > 4 ) ? 4 : oc );
    int fl_c = ( fl < 0 ) ? 0 : ( ( fl > 16 ) ? 16 : fl );
    tf[0] = (int16_t)( om_c << 3 );
    tf[1] = (int16_t)( oc_c << 6 );
    tf[2] = (int16_t)( ( tact->is_corner ? 1 : 0 ) << 7 );
    tf[3] = (int16_t)( ( tact->is_open_x_or_c ? 1 : 0 ) << 7 );
    tf[4] = (int16_t)( ( tact->quad_parity ? 1 : 0 ) << 7 );
    tf[5] = (int16_t)( fl_c << 3 );
    tf[6] = (int16_t)( ( om == 0 ? 1 : 0 ) << 7 );
    tf[7] = (int16_t)( ( om == 1 ? 1 : 0 ) << 7 );
  } else {
    for ( int k = 0; k < 8; k++ )
      tf[k] = 0;
  }

  ts[0] = (int16_t)( ( (int32_t)tf[0] * (int32_t)tf[0] ) >> 7 );
  ts[1] = (int16_t)( ( (int32_t)tf[0] * (int32_t)tf[4] ) >> 7 );
  ts[2] = (int16_t)( ( (int32_t)tf[0] * (int32_t)pf[1] ) >> 7 );
  ts[3] = (int16_t)( ( (int32_t)tf[4] * (int32_t)pf[1] ) >> 7 );
  ts[4] = (int16_t)( ( (int32_t)pf[9] * (int32_t)pf[1] ) >> 7 );
  ts[5] = (int16_t)( ( (int32_t)pf[10] * (int32_t)pf[1] ) >> 7 );
  ts[6] = (int16_t)( ( (int32_t)tf[5] * (int32_t)tf[5] ) >> 7 );
  ts[7] = (int16_t)( ( (int32_t)pf[13] * (int32_t)pf[13] ) >> 7 );
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
    a7 = vaddq_s16( a7, vld1q_s16( &wf[56] ) );
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
    for ( int k = 0; k < NNUE_ACC_DIM; k++ )
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
  int16_t vp[NNUE_CHANNELS][8];
  int16_t vn[NNUE_CHANNELS][8];
  int16_t r[NNUE_CHANNELS][8];
  int16_t a[NNUE_CHANNELS][8];
  int32_t s_vp[NNUE_CHANNELS];
  int32_t s_vn[NNUE_CHANNELS];
  int32_t s_r[NNUE_CHANNELS];
  int32_t s_a[NNUE_CHANNELS];

  for ( int c = 0; c < NNUE_CHANNELS; c++ ) {
    int32_t sum_p = 0;
    int32_t sum_n = 0;
    for ( int g = 0; g < 8; g++ ) {
      int16_t x = acc->v[c * 8 + g];
      int16_t p = ( x > 0 ) ? ( ( x > 128 ) ? 128 : x ) : 0;
      int16_t n = ( -x > 0 ) ? ( ( -x > 128 ) ? 128 : -x ) : 0;
      vp[c][g] = p;
      vn[c][g] = n;
      r[c][g] = (int16_t)( p - n );
      a[c][g] = (int16_t)( p + n );
      sum_p += p;
      sum_n += n;
    }
    s_vp[c] = sum_p;
    s_vn[c] = sum_n;
    s_r[c] = sum_p - sum_n;
    s_a[c] = sum_p + sum_n;
  }

  int32_t feats[NNUE_FEAT_DIM];
  for ( int c = 0; c < NNUE_CHANNELS; c++ ) {
    feats[c] = s_vp[c];
    feats[NNUE_CHANNELS + c] = s_vn[c];
  }

  int num_pairs = ( NNUE_CHANNELS * ( NNUE_CHANNELS + 1 ) ) / 2;
  int p_idx = 0;
  for ( int i = 0; i < NNUE_CHANNELS; i++ ) {
    for ( int j = i; j < NNUE_CHANNELS; j++ ) {
      int32_t sum_rr = 0;
      int32_t sum_aa = 0;
      int32_t sum_ra = 0;
      for ( int g = 0; g < 8; g++ ) {
        sum_rr += ( (int32_t)r[i][g] * (int32_t)r[j][g] ) >> 7;
        sum_aa += ( (int32_t)a[i][g] * (int32_t)a[j][g] ) >> 7;
        sum_ra += ( (int32_t)r[i][g] * (int32_t)a[j][g] ) >> 7;
      }
      feats[2 * NNUE_CHANNELS + 0 * num_pairs + p_idx] = sum_rr;
      feats[2 * NNUE_CHANNELS + 1 * num_pairs + p_idx] = sum_aa;
      feats[2 * NNUE_CHANNELS + 2 * num_pairs + p_idx] = sum_ra;
      feats[2 * NNUE_CHANNELS + 3 * num_pairs + p_idx] = ( s_r[i] * s_r[j] ) >> 10;
      feats[2 * NNUE_CHANNELS + 4 * num_pairs + p_idx] = ( s_a[i] * s_a[j] ) >> 10;
      p_idx++;
    }
  }

  int base_dim = 2 * NNUE_CHANNELS + 5 * num_pairs;
  int16_t pf[16];
  int16_t tf[8];
  int16_t ts[8];
  nnue_compute_parity_and_tactical( n_my, n_opp, tact, pf, tf, ts );

  for ( int k = 0; k < 16; k++ )
    feats[base_dim + k] = pf[k];
  for ( int k = 0; k < 8; k++ )
    feats[base_dim + 16 + k] = tf[k];
  for ( int k = 0; k < 8; k++ )
    feats[base_dim + 24 + k] = ts[k];

  int32_t keys[4] = {
    tf[0],
    tf[4],
    pf[1],
    (int32_t)pf[9] - (int32_t)pf[10]
  };

  for ( int c = 0; c < NNUE_CHANNELS; c++ ) {
    for ( int m = 0; m < 4; m++ ) {
      feats[base_dim + 32 + c * 4 + m] = ( s_r[c] * keys[m] ) >> 9;
      feats[base_dim + 32 + NNUE_CHANNELS * 4 + c * 4 + m] = ( s_a[c] * keys[m] ) >> 9;
    }
  }

  int b = nnue_stage_bucket( empties );
  const int16_t *ow = nnue_order_weight[b];
  const int16_t *vw = nnue_value_weight[b];
  int32_t ord_sum = nnue_order_bias[b];
  int32_t val_sum = nnue_value_bias[b];

  for ( int k = 0; k < NNUE_FEAT_DIM; k++ ) {
    ord_sum += feats[k] * (int32_t)ow[k];
    val_sum += feats[k] * (int32_t)vw[k];
  }

  if ( out_raw_order )
    *out_raw_order = ord_sum;
  if ( out_raw_value )
    *out_raw_value = val_sum;
}

#if USE_NNUE_NEON && ( NNUE_CHANNELS == 8 )
static inline int16x4_t
hsum4_s16( int16x8_t v0, int16x8_t v1, int16x8_t v2, int16x8_t v3 ) {
  int16x8_t p01 = vpaddq_s16( v0, v1 );
  int16x8_t p23 = vpaddq_s16( v2, v3 );
  int16x8_t p0123 = vpaddq_s16( p01, p23 );
  return vpadd_s16( vget_low_s16( p0123 ), vget_high_s16( p0123 ) );
}
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
#if USE_NNUE_NEON && ( NNUE_CHANNELS == 8 )
  const int16x8_t z = vdupq_n_s16( 0 );
  const int16x8_t q128 = vdupq_n_s16( 128 );

  int16x8_t r[8];
  int16x8_t a[8];
  int16_t feats[NNUE_FEAT_DIM] __attribute__((aligned(16)));
  int32_t s_r[8];
  int32_t s_a[8];

  for ( int c = 0; c < 8; c++ ) {
    int16x8_t x = vld1q_s16( &acc->v[c * 8] );
    int16x8_t vp = vminq_s16( vmaxq_s16( x, z ), q128 );
    int16x8_t vn = vminq_s16( vmaxq_s16( vnegq_s16( x ), z ), q128 );
    r[c] = vsubq_s16( vp, vn );
    a[c] = vaddq_s16( vp, vn );
    int16_t svp = vaddvq_s16( vp );
    int16_t svn = vaddvq_s16( vn );
    feats[c] = svp;
    feats[8 + c] = svn;
    s_r[c] = (int32_t)svp - (int32_t)svn;
    s_a[c] = (int32_t)svp + (int32_t)svn;
  }

  /* Process all 36 pairs in 9 batches of 4 using pairwise NEON reductions (hsum4_s16) */
  int16x8_t rr_buf[4], aa_buf[4], ra_buf[4];
  int b_cnt = 0;
  int p_idx = 0;
  for ( int i = 0; i < 8; i++ ) {
    int16x8_t ri = r[i];
    int16x8_t ai = a[i];
    int32_t sri = s_r[i];
    int32_t sai = s_a[i];
    for ( int j = i; j < 8; j++ ) {
      rr_buf[b_cnt] = vshrq_n_s16( vmulq_s16( ri, r[j] ), 7 );
      aa_buf[b_cnt] = vshrq_n_s16( vmulq_s16( ai, a[j] ), 7 );
      ra_buf[b_cnt] = vshrq_n_s16( vmulq_s16( ri, a[j] ), 7 );
      feats[16 + 3 * 36 + p_idx] = (int16_t)( ( sri * s_r[j] ) >> 10 );
      feats[16 + 4 * 36 + p_idx] = (int16_t)( ( sai * s_a[j] ) >> 10 );
      b_cnt++;
      p_idx++;
      if ( b_cnt == 4 ) {
        int base_p = p_idx - 4;
        vst1_s16( &feats[16 + 0 * 36 + base_p], hsum4_s16( rr_buf[0], rr_buf[1], rr_buf[2], rr_buf[3] ) );
        vst1_s16( &feats[16 + 1 * 36 + base_p], hsum4_s16( aa_buf[0], aa_buf[1], aa_buf[2], aa_buf[3] ) );
        vst1_s16( &feats[16 + 2 * 36 + base_p], hsum4_s16( ra_buf[0], ra_buf[1], ra_buf[2], ra_buf[3] ) );
        b_cnt = 0;
      }
    }
  }

  /* Compute 16 Corner/Quadrant Empty Parity + 8 Tactical + 8 Tactical SCReLU + 64 Cross features */
  nnue_compute_parity_and_tactical(
    n_my, n_opp, tact,
    &feats[196],
    &feats[212],
    &feats[220]
  );

  int32_t keys[4] = {
    feats[212 + 0],
    feats[212 + 4],
    feats[196 + 1],
    (int32_t)feats[196 + 9] - (int32_t)feats[196 + 10]
  };

  for ( int c = 0; c < 8; c++ ) {
    int32_t src = s_r[c];
    int32_t sac = s_a[c];
    feats[228 + c * 4 + 0] = (int16_t)( ( src * keys[0] ) >> 9 );
    feats[228 + c * 4 + 1] = (int16_t)( ( src * keys[1] ) >> 9 );
    feats[228 + c * 4 + 2] = (int16_t)( ( src * keys[2] ) >> 9 );
    feats[228 + c * 4 + 3] = (int16_t)( ( src * keys[3] ) >> 9 );
    feats[260 + c * 4 + 0] = (int16_t)( ( sac * keys[0] ) >> 9 );
    feats[260 + c * 4 + 1] = (int16_t)( ( sac * keys[1] ) >> 9 );
    feats[260 + c * 4 + 2] = (int16_t)( ( sac * keys[2] ) >> 9 );
    feats[260 + c * 4 + 3] = (int16_t)( ( sac * keys[3] ) >> 9 );
  }

  int b = nnue_stage_bucket( empties );
  const int16_t *ow = nnue_order_weight[b];
  const int16_t *vw = nnue_value_weight[b];

  if ( out_raw_order ) {
    int32x4_t acc_o0 = vdupq_n_s32( 0 );
    int32x4_t acc_o1 = vdupq_n_s32( 0 );
    int k = 0;
    for ( ; k + 8 <= NNUE_FEAT_DIM; k += 8 ) {
      int16x8_t fk = vld1q_s16( &feats[k] );
      int16x8_t wk = vld1q_s16( &ow[k] );
      acc_o0 = vmlal_s16( acc_o0, vget_low_s16( fk ), vget_low_s16( wk ) );
      acc_o1 = vmlal_s16( acc_o1, vget_high_s16( fk ), vget_high_s16( wk ) );
    }
    int32_t ord_sum = nnue_order_bias[b] + vaddvq_s32( vaddq_s32( acc_o0, acc_o1 ) );
    for ( ; k < NNUE_FEAT_DIM; k++ )
      ord_sum += (int32_t)feats[k] * (int32_t)ow[k];
    *out_raw_order = ord_sum;
  }

  if ( out_raw_value ) {
    int32x4_t acc_v0 = vdupq_n_s32( 0 );
    int32x4_t acc_v1 = vdupq_n_s32( 0 );
    int k = 0;
    for ( ; k + 8 <= NNUE_FEAT_DIM; k += 8 ) {
      int16x8_t fk = vld1q_s16( &feats[k] );
      int16x8_t wk = vld1q_s16( &vw[k] );
      acc_v0 = vmlal_s16( acc_v0, vget_low_s16( fk ), vget_low_s16( wk ) );
      acc_v1 = vmlal_s16( acc_v1, vget_high_s16( fk ), vget_high_s16( wk ) );
    }
    int32_t val_sum = nnue_value_bias[b] + vaddvq_s32( vaddq_s32( acc_v0, acc_v1 ) );
    for ( ; k < NNUE_FEAT_DIM; k++ )
      val_sum += (int32_t)feats[k] * (int32_t)vw[k];
    *out_raw_value = val_sum;
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
nnue_evaluate_board_value( uint64_t my_bits, uint64_t opp_bits, int empties ) {
  NnueAcc acc;
  nnue_init_acc( &acc, my_bits, opp_bits );
  return nnue_evaluate_value_acc( &acc, my_bits, opp_bits, 0, empties );
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
