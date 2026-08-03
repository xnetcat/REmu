/* Copyright (C) 2026 REmu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Thin in-process libavformat/libavcodec bridge for AvPlayer + Videodec2 (plan P5).
 * System ffmpeg is linked at build time when available for the host target.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RemuLibavDemux RemuLibavDemux;
typedef struct RemuLibavCodec RemuLibavCodec;

/* Probe helpers (no open session retained). Returns 1 on success, 0 on failure. */
int remu_libav_probe_video(
    const char *path,
    int32_t *width_out,
    int32_t *height_out,
    double *fps_out,
    double *duration_out);
int remu_libav_probe_has_audio(const char *path);

/*
 * Open a demux+decode session for one stream kind.
 * audio=0 → packed NV12 frames of size width*height*3/2
 * audio=1 → interleaved s16le stereo @ 48000 Hz, frame_bytes samples worth
 * seek_seconds ≥ 0 seeks before the first frame.
 * Returns NULL on failure and writes a short message into err_buf when provided.
 */
RemuLibavDemux *remu_libav_demux_open(
    const char *path,
    int audio,
    double seek_seconds,
    char *err_buf,
    size_t err_buf_len);

/* 0 = frame written, 1 = EOF, -1 = error. out_len must match the packed frame size. */
int remu_libav_demux_read_frame(RemuLibavDemux *session, uint8_t *out, size_t out_len);

void remu_libav_demux_close(RemuLibavDemux *session);

/*
 * Elementary-stream codec session for Videodec2 (AVC/HEVC).
 * codec_id: 1 = AVC, 2 = HEVC (matches common Orbis Videodec2 codecType values).
 */
RemuLibavCodec *remu_libav_codec_open(uint32_t codec_id, char *err_buf, size_t err_buf_len);
int remu_libav_codec_send(
    RemuLibavCodec *codec,
    const uint8_t *au,
    size_t au_size,
    int64_t pts);
/* 0 = frame, 1 = need more input, -1 = error. Writes packed NV12; sets width/height/pitch outs. */
int remu_libav_codec_receive_nv12(
    RemuLibavCodec *codec,
    uint8_t *out,
    size_t out_len,
    int32_t *width_out,
    int32_t *height_out,
    int32_t *pitch_out);
int remu_libav_codec_flush(RemuLibavCodec *codec);
void remu_libav_codec_close(RemuLibavCodec *codec);

/* 1 when the bridge was built against real libav*, 0 for the compile stub. */
int remu_libav_available(void);

#ifdef __cplusplus
}
#endif
